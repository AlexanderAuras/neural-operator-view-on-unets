import argparse
import os
from pathlib import Path
import random
import sys
import zipfile

import numpy as np
import randomname
import torch
from torch import nn
import torch.backends.cudnn
from torch.utils.data import DataLoader
import torch.utils.tensorboard
import torchinfo
import torchview
from tqdm.auto import tqdm, trange

from fno_unet.classical_unet import UNet
from fno_unet.ct_dataset import CTPostProcessDataset
from fno_unet.ellipses_dataset import EllipsesDataset
from fno_unet.fno_unet import FNOUNet
from fno_unet.interp_unet import InterpolatingUNet


OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / randomname.get_name()


def __init_worker(_: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def main() -> None:
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--seed", type=int, default=random.randrange(0, 2**32))
    argparser.add_argument("--device", type=torch.device, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    argparser.add_argument("--num-workers", type=int, default=os.cpu_count())
    argparser.add_argument("--no-compile", dest="compile", action="store_false")
    argparser.add_argument("--precision", choices=["high", "medium", "low"], default="medium")
    argparser.add_argument("--dataset", choices=["LoDoPaB"], required=True)
    argparser.add_argument("--model", choices=["Classic", "Interp", "FNO"], required=True)
    argparser.add_argument("--batch-size", type=int, default=32)
    argparser.add_argument("--max-epochs", type=int, default=10)
    argparser.add_argument("--lr", type=float, default=1e-3)
    argparser.add_argument("--model-save-freq", type=int, default=1)
    argparser.add_argument("--noise-level", type=float, default=100.0)
    args = argparser.parse_args()

    OUT_DIR.mkdir(parents=True)

    with zipfile.ZipFile(OUT_DIR / "code.zip", "w") as code_archive:
        for file in Path(__file__).parents[1].resolve().glob("ipnas/**/*.py"):
            code_archive.write(file, file.relative_to(Path(__file__).parents[1].resolve()))

    # Configure PyTorch
    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.float64 if args.precision == "high" else torch.float32)
    torch.set_float32_matmul_precision({"high": "highest", "medium": "high", "low": "medium"}[args.precision])
    torch.set_anomaly_enabled(args.debug)
    torch.autograd.set_detect_anomaly(args.debug)  # pyright: ignore [reportPrivateImportUsage]  # BUG Invalid pyright warning

    # Setup deterministic behavior
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Load data
    match args.dataset:
        # case "CIFAR10":
        #     trainval_dataset = cast(Dataset[dict[str, Any]], datasets.load_dataset("CIFAR10", split="train").with_format("torch"))
        #     train_len = round(0.8 * len(cast(Sized, trainval_dataset)))
        #     val_len = len(cast(Sized, trainval_dataset)) - train_len
        #     train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_len, val_len])
        #     test_dataset = cast(Dataset[dict[str, Any]], datasets.load_dataset("CIFAR10", split="test").with_format("torch"))
        #     dataset_shapes = {"input": (3, 32, 32), "target": (1,)}
        case "LoDoPaB":
            train_dataset = CTPostProcessDataset(EllipsesDataset(6400, 256, 10), args.noise_level)
            val_dataset = CTPostProcessDataset(EllipsesDataset(1600, 256, 10), args.noise_level)
            test_dataset = CTPostProcessDataset(EllipsesDataset(2000, 256, 10), args.noise_level)
            image_shape = (1, 256, 256)
        case _:
            raise ValueError(f'Unknown dataset: "{args.dataset}"')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=__init_worker)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=__init_worker)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=__init_worker)

    # Create model and log model information
    match args.model:
        case "UNet":
            model = UNet(1, 1).to(args.device)
        case "Interp":
            model = InterpolatingUNet(1, 1, base_input_size=128, max_scale_factor=8).to(args.device)
        case "FNO":
            model = FNOUNet(1, 1).to(args.device)
        case _:
            raise ValueError(f'Unknown model: "{args.model}"')
    fwd_func = torch.compile(model) if args.compile else model.__call__
    compute_graph = torchview.draw_graph(model, input_size=(1, *image_shape), show_shapes=True, show_ops=True, show_attrs=True, show_coloring=True)
    compute_graph.visual_graph.render("compute-graph", directory=OUT_DIR, format="pdf")
    OUT_DIR.joinpath("compute-graph").unlink()
    model_statistics = torchinfo.summary(model, input_size=(1, *image_shape), verbose=0)
    print(model_statistics)

    # Create loss function, optimizer, and learning rate scheduler
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    tb_logger = torch.utils.tensorboard.writer.SummaryWriter(OUT_DIR)
    torch.save(model.state_dict(), OUT_DIR / "initial.pt")
    torch.save(model.state_dict(), OUT_DIR / "best.pt")

    # Initial validation before training
    model.eval()
    val_loss = 0.0
    for sample in tqdm(val_dataloader, desc="Validation", unit="batch", position=1, leave=False):
        input_ = sample["input"].to(args.device)
        target = sample["target"].to(args.device)
        prediction = fwd_func(input_)
        val_loss += loss_function(prediction, target).item()
    val_loss /= len(val_dataloader)
    best_val_loss = val_loss
    tb_logger.add_scalar("val/loss", val_loss, global_step=0)

    # Main training loop
    try:
        epoch = 0
        epochs_iter = trange(args.max_epochs, desc="Epochs", unit="epoch", position=0, leave=False)
        for epoch in epochs_iter:
            # Train one epoch
            model.train()
            batches_iter = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training", unit="batch", position=1, leave=False)
            for batch_no, sample in batches_iter:
                input_ = sample["input"].to(args.device)
                target = sample["target"].to(args.device)
                with torch.enable_grad():
                    prediction = fwd_func(input_)
                    loss = loss_function(prediction, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tb_logger.add_scalar("train/loss", loss.item(), global_step=epoch * len(train_dataloader) + batch_no)
                batches_iter.set_postfix({"Train-Loss": loss.item()})
            batches_iter.close()

            # Validate model
            model.eval()
            val_loss = 0.0
            for sample in tqdm(val_dataloader, desc="Validation", unit="batch", position=1, leave=False):
                input_ = sample["input"].to(args.device)
                target = sample["target"].to(args.device)
                prediction = fwd_func(input_)
                val_loss += loss_function(prediction, target).item()
            val_loss /= len(val_dataloader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), OUT_DIR / "best.pt")
            tb_logger.add_scalar("val/loss", val_loss, global_step=(epoch + 1) * len(train_dataloader))
            lr_scheduler.step()
            epochs_iter.set_postfix({"Val-Loss": val_loss})

            # Save model
            if (epoch + 1) % args.model_log_freq == 0:
                torch.save(model.state_dict(), OUT_DIR / f"epoch-{epoch + 1}.pt")
        epochs_iter.close()
    except KeyboardInterrupt:
        pass

    # Save model after training
    torch.save(model.state_dict(), OUT_DIR / "final.pt")
    torch.onnx.export(model, (torch.randn(image_shape),), str(OUT_DIR / "model.onnx"))

    # Test model and log test performance
    model.load_state_dict(torch.load(OUT_DIR / "best.pt"))
    model.eval()
    test_loss = 0.0
    for sample in test_dataloader:
        input_ = sample["input"].to(args.device)
        target = sample["target"].to(args.device)
        prediction = fwd_func(input_)
        test_loss += loss_function(prediction, target).item()
    test_loss /= len(test_dataloader)
    print(f"Final Test-Loss: {test_loss}")
    tb_logger.add_scalar("test/loss", test_loss, global_step=0)

    # Cleanup
    tb_logger.flush()
    tb_logger.close()


if __name__ == "__main__":
    sys.exit(main())
