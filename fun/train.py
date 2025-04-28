import argparse
from math import ceil
import os
from pathlib import Path
import random
import shutil
import sys
import warnings
import zipfile

import numpy as np
import randomname
import torch
from torch import nn
import torch.backends.cudnn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.tensorboard
import torchinfo
import torchview
from tqdm.auto import tqdm, trange

from fun.classical_unet import UNet
from fun.ct_dataset import CTPostProcessDataset
from fun.ellipses_dataset import EllipsesDataset
from fun.fno_unet import FNOUNet
from fun.interp_unet import InterpolatingUNet
from fun.multi_res_batch_sampler import MultiResolutionBatchSampler


OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / randomname.get_name()


def __init_worker(_: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def norm(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def main() -> None:
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--seed", type=int, default=random.randrange(0, 2**32))
    argparser.add_argument("--device", type=torch.device, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    argparser.add_argument("--num-workers", type=int, default=os.cpu_count())
    argparser.add_argument("--no-compile", dest="compile", action="store_false")
    argparser.add_argument("--precision", choices=["high", "medium", "low"], default="medium")
    argparser.add_argument("--dataset", choices=["ellipses-64x64", "ellipses-128x128", "ellipses-256x256", "ellipses-mixed"], required=True)
    argparser.add_argument("--model", choices=["classic", "interp", "fno"], required=True)
    argparser.add_argument("--batch-size", type=int, default=32)
    argparser.add_argument("--max-epochs", type=int, default=10)
    argparser.add_argument("--lr", type=float, default=1e-3)
    argparser.add_argument("--model-save-freq", type=int, default=1)
    argparser.add_argument("--noise-level", type=float, default=0.0)
    argparser.add_argument("--angle-percent", type=float, default=0.75)
    args = argparser.parse_args()

    if args.debug:
        globals()["OUT_DIR"] = OUT_DIR.parent.joinpath("_debug")
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    print("=" * 30 + " " + OUT_DIR.name.upper() + " " + "=" * 30)

    with zipfile.ZipFile(OUT_DIR / "code.zip", "w") as code_archive:
        for file in Path(__file__).parents[1].resolve().glob("fun/**/*.py"):
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
        case "ellipses-64x64":
            train_dataset = CTPostProcessDataset(
                EllipsesDataset(1600, 1024, 10),
                angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                pos_count=128,
                target_shape=(64, 64),
                noise_type="gaussian",
                noise_level=args.noise_level,
            )
            train_batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(train_dataset), batch_size=args.batch_size, drop_last=False)
            val_datasets = {
                "64x64": CTPostProcessDataset(
                    EllipsesDataset(400, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                    pos_count=128,
                    target_shape=(64, 64),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
            }
            test_datasets = {
                "64x64": CTPostProcessDataset(
                    EllipsesDataset(500, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                    pos_count=128,
                    target_shape=(64, 64),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
            }
            exemplary_image_shape = (1, 64, 64)
        case "ellipses-128x128":
            train_dataset = CTPostProcessDataset(
                EllipsesDataset(1600, 1024, 10),
                angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                pos_count=256,
                target_shape=(128, 128),
                noise_type="gaussian",
                noise_level=args.noise_level,
            )
            train_batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(train_dataset), batch_size=args.batch_size, drop_last=False)
            val_datasets = {
                "128x128": CTPostProcessDataset(
                    EllipsesDataset(400, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                    pos_count=256,
                    target_shape=(128, 128),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
            }
            test_datasets = {
                "128x128": CTPostProcessDataset(
                    EllipsesDataset(500, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                    pos_count=256,
                    target_shape=(128, 128),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
            }
            exemplary_image_shape = (1, 128, 128)
        case "ellipses-256x256":
            train_dataset = CTPostProcessDataset(
                EllipsesDataset(1600, 1024, 10),
                angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                pos_count=512,
                target_shape=(256, 256),
                noise_type="gaussian",
                noise_level=args.noise_level,
            )
            train_batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(train_dataset), batch_size=args.batch_size, drop_last=False)
            val_datasets = {
                "256x256": CTPostProcessDataset(
                    EllipsesDataset(400, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                    pos_count=512,
                    target_shape=(256, 256),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
            }
            test_datasets = {
                "256x256": CTPostProcessDataset(
                    EllipsesDataset(500, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                    pos_count=512,
                    target_shape=(256, 256),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
            }
            exemplary_image_shape = (1, 256, 256)
        case "ellipses-mixed":
            train_dataset = torch.utils.data.ConcatDataset(
                [
                    CTPostProcessDataset(
                        EllipsesDataset(1600, 1024, 10),
                        angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                        pos_count=128,
                        target_shape=(64, 64),
                        noise_type="gaussian",
                        noise_level=args.noise_level,
                    ),
                    CTPostProcessDataset(
                        EllipsesDataset(1600, 1024, 10),
                        angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                        pos_count=256,
                        target_shape=(128, 128),
                        noise_type="gaussian",
                        noise_level=args.noise_level,
                    ),
                    CTPostProcessDataset(
                        EllipsesDataset(1600, 1024, 10),
                        angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                        pos_count=512,
                        target_shape=(256, 256),
                        noise_type="gaussian",
                        noise_level=args.noise_level,
                    ),
                ]
            )
            train_batch_sampler = MultiResolutionBatchSampler([1600] * 3, batch_size=args.batch_size, shuffle=True, drop_incomplete=False)
            val_datasets = {
                "64x64": CTPostProcessDataset(
                    EllipsesDataset(400, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                    pos_count=128,
                    target_shape=(64, 64),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
                "128x128": CTPostProcessDataset(
                    EllipsesDataset(400, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                    pos_count=256,
                    target_shape=(128, 128),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
                "256x256": CTPostProcessDataset(
                    EllipsesDataset(400, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                    pos_count=512,
                    target_shape=(256, 256),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
            }
            test_datasets = {
                "64x64": CTPostProcessDataset(
                    EllipsesDataset(500, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                    pos_count=128,
                    target_shape=(64, 64),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
                "128x128": CTPostProcessDataset(
                    EllipsesDataset(500, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                    pos_count=256,
                    target_shape=(128, 128),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
                "256x256": CTPostProcessDataset(
                    EllipsesDataset(500, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                    pos_count=512,
                    target_shape=(256, 256),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
            }
            exemplary_image_shape = (1, 256, 256)
        case _:
            raise ValueError(f'Unknown dataset: "{args.dataset}"')
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        worker_init_fn=__init_worker,
    )
    val_dataloaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            worker_init_fn=__init_worker,
        )
        for name, dataset in val_datasets.items()
    }
    test_dataloaders = {
        name: DataLoader(
            dataset,
            batch_sampler=MultiResolutionBatchSampler([500] * 3, batch_size=args.batch_size, shuffle=False, drop_incomplete=False),
            num_workers=args.num_workers,
            worker_init_fn=__init_worker,
        )
        for name, dataset in test_datasets.items()
    }

    # Create model and log model information
    match args.model:
        case "classic":
            model = UNet(1, 1).to(args.device)
        case "interp":
            model = InterpolatingUNet(1, 1, base_input_size=128, max_scale_factor=8).to(args.device)
        case "fno":
            model = FNOUNet(1, 1).to(args.device)
        case _:
            raise ValueError(f'Unknown model: "{args.model}"')
    fwd_func = torch.compile(model) if args.compile else model.__call__
    compute_graph = torchview.draw_graph(model, input_size=(1, *exemplary_image_shape), show_shapes=True, device=args.device)
    compute_graph.visual_graph.render("compute-graph", directory=OUT_DIR, format="pdf")
    OUT_DIR.joinpath("compute-graph").unlink()
    model_statistics = torchinfo.summary(model, input_size=(1, *exemplary_image_shape), verbose=0, device=args.device)
    print(model_statistics)

    # Create loss function, optimizer, and learning rate scheduler
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    tb_logger = torch.utils.tensorboard.writer.SummaryWriter(OUT_DIR)
    torch.save(model.state_dict(), OUT_DIR / "initial.pt")
    [torch.save(model.state_dict(), OUT_DIR / f"best-{name}.pt") for name in val_dataloaders.keys()]
    best_val_losses = {name: float("inf") for name in val_dataloaders.keys()}

    # Initial validation before training
    model.eval()
    for name, dataloader in tqdm(val_dataloaders.items(), desc="Validation", unit="dataset", position=0, leave=False):
        val_loss = 0.0
        for i, sample in tqdm(enumerate(dataloader), total=len(dataloader), desc=name, unit="batch", position=1, leave=False):
            input_ = sample["input"].to(args.device)
            target = sample["target"].to(args.device)
            prediction = fwd_func(input_)
            val_loss += loss_function(prediction, target).item()
            if i == 0:
                [tb_logger.add_image(f"val/{name}-input-{j}", norm(input_[j]), global_step=0) for j in range(min(4, input_.shape[0]))]
                [tb_logger.add_image(f"val/{name}-target-{j}", norm(target[j]), global_step=0) for j in range(min(4, target.shape[0]))]
                [tb_logger.add_image(f"val/{name}-prediction-{j}", norm(prediction[j]), global_step=0) for j in range(min(4, prediction.shape[0]))]
        val_loss /= len(dataloader)
        best_val_losses[name] = val_loss
        tb_logger.add_scalar(f"val/{name}-loss", val_loss, global_step=0)

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
            for name, dataloader in tqdm(val_dataloaders.items(), desc="Validation", unit="dataset", position=1, leave=False):
                val_loss = 0.0
                for i, sample in tqdm(enumerate(dataloader), total=len(dataloader), desc=name, unit="batch", position=2, leave=False):
                    input_ = sample["input"].to(args.device)
                    target = sample["target"].to(args.device)
                    prediction = fwd_func(input_)
                    val_loss += loss_function(prediction, target).item()
                    if i == 0:
                        [tb_logger.add_image(f"val/{name}-input-{j}", norm(input_[j]), global_step=epoch + 1) for j in range(min(4, input_.shape[0]))]
                        [tb_logger.add_image(f"val/{name}-target-{j}", norm(target[j]), global_step=epoch + 1) for j in range(min(4, target.shape[0]))]
                        [tb_logger.add_image(f"val/{name}-prediction-{j}", norm(prediction[j]), global_step=epoch + 1) for j in range(min(4, prediction.shape[0]))]
                val_loss /= len(dataloader)
                if val_loss < best_val_losses[name]:
                    best_val_losses[name] = val_loss
                    torch.save(model.state_dict(), OUT_DIR / f"best-{name}.pt")
                tb_logger.add_scalar(f"val/{name}-loss", val_loss, global_step=epoch + 1)
                lr_scheduler.step()

            # Save model
            if (epoch + 1) % args.model_save_freq == 0:
                torch.save(model.state_dict(), OUT_DIR / f"epoch-{epoch + 1}.pt")
        epochs_iter.close()
    except KeyboardInterrupt:
        pass

    # Save model after training
    torch.save(model.state_dict(), OUT_DIR / "final.pt")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=r"Converting a tensor to a Python boolean might cause the trace to be incorrect.*")
    torch.onnx.export(model, (torch.randn((1, *exemplary_image_shape), device=args.device),), str(OUT_DIR / "model.onnx"))

    # Test model and log test performance
    model.eval()
    for name, dataloader in tqdm(test_dataloaders.items(), desc="Testing", unit="dataset", position=0, leave=True):
        test_loss = 0.0
        model.load_state_dict(torch.load(OUT_DIR / f"best-{name}.pt"))
        for i, sample in tqdm(enumerate(dataloader), total=len(dataloader), desc=name, unit="batch", position=1, leave=False):
            input_ = sample["input"].to(args.device)
            target = sample["target"].to(args.device)
            prediction = fwd_func(input_)
            test_loss += loss_function(prediction, target).item()
            if i == 0:
                [tb_logger.add_image(f"test/{name}-input-{j}", norm(input_[j]), global_step=0) for j in range(min(4, input_.shape[0]))]
                [tb_logger.add_image(f"test/{name}-target-{j}", norm(target[j]), global_step=0) for j in range(min(4, target.shape[0]))]
                [tb_logger.add_image(f"test/{name}-prediction-{j}", norm(prediction[j]), global_step=0) for j in range(min(4, prediction.shape[0]))]
        test_loss /= len(dataloader)
        print(f'Final Test-Loss on "{name}": {test_loss}')
        tb_logger.add_scalar(f"test/{name}-loss", test_loss, global_step=0)

    # Cleanup
    tb_logger.flush()
    tb_logger.close()


if __name__ == "__main__":
    sys.exit(main())
