import argparse
from contextlib import redirect_stderr, redirect_stdout
import datetime
import logging
import logging.config
from math import ceil
import os
from pathlib import Path
import random
import shutil
import sys
from typing import cast
import warnings
import zipfile

import numpy as np
import PIL.Image as Image
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
from fun.fno_unet import FNOUNet, HeatUNet
from fun.interp_unet import InterpolatingUNet
from fun.multi_res_batch_sampler import MultiResolutionBatchSampler


def setup_logging(logging_config_path: str | Path, out_dir: str | Path | None = None) -> None:
    try:
        logging.config.fileConfig(Path(logging_config_path).resolve())
        if out_dir is not None:
            for logger in [logging.getLogger(name) for name in [*logging.root.manager.loggerDict.keys(), None]]:
                for handler in logger.handlers:
                    if isinstance(handler, logging.FileHandler):
                        handler.baseFilename = str(Path(out_dir) / Path(handler.baseFilename).name)
                        print(handler.baseFilename)
    except ModuleNotFoundError as e:
        indent_depth = 46 + len(str(logging_config_path))
        indented_error_msg = " " * indent_depth + str(e).replace("\n", "\n" + " " * indent_depth)
        print(f'Failed to load logging configuration from "{logging_config_path}": {type(e).__name__}\n{indented_error_msg}', file=sys.stderr)
        exit(-1)
    old_showwarning = warnings.showwarning
    warnings.showwarning = lambda m, c, fn, ln, f, l: old_showwarning(m, c, fn, ln, f, l) if f is not None else logging.getLogger("py.warnings").warning(m)  # noqa: E741


def __init_worker(_: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def normalized(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def main() -> None:
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--seed", type=int, default=random.randrange(0, 2**32))
    argparser.add_argument("--device", type=torch.device, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    argparser.add_argument("--num-workers", type=int, default=os.cpu_count())
    argparser.add_argument("--no-compile", dest="compile", action="store_false")
    argparser.add_argument("--data-parallel", action="store_true")
    argparser.add_argument("--precision", choices=["high", "medium", "low"], default="medium")
    argparser.add_argument("--dataset", choices=["ellipses-64x64", "ellipses-128x128", "ellipses-256x256", "ellipses-mixed"], required=True)
    argparser.add_argument("--model", choices=["classic", "interp", "fno", "heat"], required=True)
    argparser.add_argument("--batch-size", type=int, default=32)
    argparser.add_argument("--max-epochs", type=int, default=10)
    argparser.add_argument("--lr", type=float, default=1e-3)
    argparser.add_argument("--model-save-freq", type=int, default=1)
    argparser.add_argument("--noise-level", type=float, default=0.0)
    argparser.add_argument("--angle-percent", type=float, default=0.75)
    args = argparser.parse_args()

    out_dir = Path(__file__).resolve().parents[1] / "runs" / randomname.get_name()
    if args.debug:
        out_dir = out_dir.parent.joinpath("_debug")
        if out_dir.exists():
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    if "SLURM_JOB_ID" in os.environ:
        redirect_stdout(out_dir.joinpath("stdout.log").open("w")).__enter__()
        redirect_stderr(out_dir.joinpath("stderr.log").open("w")).__enter__()
    setup_logging(Path(__file__).resolve().parents[1] / "logging.conf", out_dir)
    logger = logging.getLogger("fun.train")
    logger.info(f"{'=' * 20} {out_dir.name.upper()} {'=' * 20}")
    logger.debug("Command line arguments:")
    for arg, value in vars(args).items():
        logger.debug(f"    {arg}: {value}")
    logger.info(f"Output directory: {out_dir}")

    logger.info("Saving code to archive")
    with zipfile.ZipFile(out_dir / "code.zip", "w") as code_archive:
        for file in Path(__file__).parents[1].resolve().glob("fun/**/*.py"):
            logger.debug(f"    Adding {file} to code-archive")
            code_archive.write(file, file.relative_to(Path(__file__).parents[1].resolve()))

    # Configure PyTorch
    logger.info("Configuring PyTorch")
    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.float64 if args.precision == "high" else torch.float32)
    torch.set_float32_matmul_precision({"high": "highest", "medium": "high", "low": "medium"}[args.precision])
    torch.set_anomaly_enabled(args.debug)
    torch.autograd.set_detect_anomaly(args.debug)  # pyright: ignore [reportPrivateImportUsage]  # BUG Invalid pyright warning

    # Setup deterministic behavior
    logger.info(f"Setting up deterministic behavior (seed: {args.seed})")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Load data
    logger.info("Loading data")
    logger.info("Loading datasets")
    match args.dataset:
        case "ellipses-64x64":
            train_dataset = CTPostProcessDataset(
                EllipsesDataset(6400, 1024, 10),
                angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                pos_count=128,
                target_shape=(64, 64),
                noise_type="gaussian",
                noise_level=args.noise_level,
            )
            train_batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(train_dataset), batch_size=args.batch_size, drop_last=False)
            val_datasets = {
                "64x64": CTPostProcessDataset(
                    EllipsesDataset(1600, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                    pos_count=128,
                    target_shape=(64, 64),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
            }
            test_dataset = EllipsesDataset(2000, 1024, 10)
            test_datasets = {
                "64x64": CTPostProcessDataset(
                    test_dataset,
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                    pos_count=128,
                    target_shape=(64, 64),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
                "128x128": CTPostProcessDataset(
                    test_dataset,
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                    pos_count=256,
                    target_shape=(128, 128),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
                "256x256": CTPostProcessDataset(
                    test_dataset,
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                    pos_count=512,
                    target_shape=(256, 256),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
            }
            exemplary_image_shape = (1, 64, 64)
        case "ellipses-128x128":
            train_dataset = CTPostProcessDataset(
                EllipsesDataset(6400, 1024, 10),
                angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                pos_count=256,
                target_shape=(128, 128),
                noise_type="gaussian",
                noise_level=args.noise_level,
            )
            train_batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(train_dataset), batch_size=args.batch_size, drop_last=False)
            val_datasets = {
                "128x128": CTPostProcessDataset(
                    EllipsesDataset(1600, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                    pos_count=256,
                    target_shape=(128, 128),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
            }
            test_dataset = EllipsesDataset(2000, 1024, 10)
            test_datasets = {
                "64x64": CTPostProcessDataset(
                    test_dataset,
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                    pos_count=128,
                    target_shape=(64, 64),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
                "128x128": CTPostProcessDataset(
                    test_dataset,
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                    pos_count=256,
                    target_shape=(128, 128),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
                "256x256": CTPostProcessDataset(
                    test_dataset,
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                    pos_count=512,
                    target_shape=(256, 256),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
            }
            exemplary_image_shape = (1, 128, 128)
        case "ellipses-256x256":
            train_dataset = CTPostProcessDataset(
                EllipsesDataset(6400, 1024, 10),
                angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                pos_count=512,
                target_shape=(256, 256),
                noise_type="gaussian",
                noise_level=args.noise_level,
            )
            train_batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(train_dataset), batch_size=args.batch_size, drop_last=False)
            val_datasets = {
                "256x256": CTPostProcessDataset(
                    EllipsesDataset(1600, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                    pos_count=512,
                    target_shape=(256, 256),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
            }
            test_dataset = EllipsesDataset(2000, 1024, 10)
            test_datasets = {
                "64x64": CTPostProcessDataset(
                    test_dataset,
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                    pos_count=128,
                    target_shape=(64, 64),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
                "128x128": CTPostProcessDataset(
                    test_dataset,
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                    pos_count=256,
                    target_shape=(128, 128),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
                "256x256": CTPostProcessDataset(
                    test_dataset,
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                    pos_count=512,
                    target_shape=(256, 256),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
            }
            exemplary_image_shape = (1, 256, 256)
        case "ellipses-mixed":
            train_datasets = [
                CTPostProcessDataset(
                    EllipsesDataset(2133, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                    pos_count=128,
                    target_shape=(64, 64),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
                CTPostProcessDataset(
                    EllipsesDataset(2133, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                    pos_count=256,
                    target_shape=(128, 128),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
                CTPostProcessDataset(
                    EllipsesDataset(2133, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                    pos_count=512,
                    target_shape=(256, 256),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
            ]
            train_dataset = torch.utils.data.ConcatDataset(train_datasets)
            train_batch_sampler = MultiResolutionBatchSampler([len(x) for x in train_datasets], batch_size=args.batch_size, shuffle=True, drop_incomplete=False)
            val_datasets = {
                "64x64": CTPostProcessDataset(
                    EllipsesDataset(533, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                    pos_count=128,
                    target_shape=(64, 64),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
                "128x128": CTPostProcessDataset(
                    EllipsesDataset(533, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                    pos_count=256,
                    target_shape=(128, 128),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
                "256x256": CTPostProcessDataset(
                    EllipsesDataset(533, 1024, 10),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                    pos_count=512,
                    target_shape=(256, 256),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
            }
            test_dataset = EllipsesDataset(2000, 1024, 10)
            test_datasets = {
                "64x64": CTPostProcessDataset(
                    test_dataset,
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                    pos_count=128,
                    target_shape=(64, 64),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
                "128x128": CTPostProcessDataset(
                    test_dataset,
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                    pos_count=256,
                    target_shape=(128, 128),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                ),
                "256x256": CTPostProcessDataset(
                    test_dataset,
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
    logger.info("Creating dataloaders")
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
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            worker_init_fn=__init_worker,
        )
        for name, dataset in test_datasets.items()
    }

    # Create model and log model information
    logger.info("Creating model")
    match args.model:
        case "classic":
            model = UNet(1, 1).to(args.device)
        case "interp":
            model = InterpolatingUNet(1, 1, base_input_size=128, max_scale_factor=8).to(args.device)
        case "fno":
            model = FNOUNet(1, 1).to(args.device)
        case "heat":
            model = HeatUNet(1).to(args.device)
        case _:
            raise ValueError(f'Unknown model: "{args.model}"')
    if args.data_parallel:
        logger.info("Parallelizing model")
        model = cast(nn.Module, nn.DataParallel(model))
    if args.compile:
        logger.info("Compiling model")
        fwd_func = torch.compile(model)
    else:
        fwd_func = model.__call__
    logger.info("Rendering compute graph")
    compute_graph = torchview.draw_graph(model, input_size=(1, *exemplary_image_shape), show_shapes=True, device=args.device)
    compute_graph.visual_graph.render("compute-graph", directory=out_dir, format="pdf")
    out_dir.joinpath("compute-graph").unlink()
    logger.info("Calculating model summary")
    model_statistics = torchinfo.summary(model, input_size=(1, *exemplary_image_shape), verbose=0, device=args.device)
    logger.info(model_statistics)

    # Create loss function, optimizer, and learning rate scheduler
    logger.info("Creating loss function, optimizer, and learning rate scheduler")
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    logger.info("Creating tensorboard logger")
    tb_logger = torch.utils.tensorboard.writer.SummaryWriter(out_dir)
    out_dir.joinpath("weights").mkdir(parents=True)
    logger.info("Saving initial weights")
    logger.debug(f"    Path: {out_dir / 'weights' / 'initial.pt'}")
    [logger.debug(f"    Path: {out_dir / 'weights' / f'best-{name}'}") for name in val_dataloaders.keys()]
    logger.debug(f"    Path: {out_dir / 'weights' / 'best-all'}")
    torch.save(model.state_dict(), out_dir / "weights" / "initial.pt")
    [torch.save(model.state_dict(), out_dir / "weights" / f"best-{name}.pt") for name in val_dataloaders.keys()]
    torch.save(model.state_dict(), out_dir / "weights" / "best-all.pt")
    best_val_losses = {name: float("inf") for name in val_dataloaders.keys()}

    # Initial validation before training
    model.eval()
    logger.info("Performing initial validation")
    for name, dataloader in tqdm(val_dataloaders.items(), desc="Validation", unit="dataset", position=0, leave=False):
        val_loss = 0.0
        for i, sample in tqdm(enumerate(dataloader), total=len(dataloader), desc=name, unit="batch", position=1, leave=False):
            input_ = sample["input"].to(args.device)
            target = sample["target"].to(args.device)
            prediction = fwd_func(input_)
            val_loss += loss_function(prediction, target).item()
            if i == 0:
                for j in range(min(4, input_.shape[0])):
                    out_dir.joinpath("val-imgs", f"{name}-input-{j}").mkdir(parents=True, exist_ok=True)
                    out_dir.joinpath("val-imgs", f"{name}-target-{j}").mkdir(parents=True, exist_ok=True)
                    out_dir.joinpath("val-imgs", f"{name}-prediction-{j}").mkdir(parents=True, exist_ok=True)
                    np.save(out_dir / "val-imgs" / f"{name}-input-{j}" / "step-0.npy", input_[j].cpu().detach().numpy())
                    np.save(out_dir / "val-imgs" / f"{name}-target-{j}" / "step-0.npy", target[j].cpu().detach().numpy())
                    np.save(out_dir / "val-imgs" / f"{name}-prediction-{j}" / "step-0.npy", prediction[j].cpu().detach().numpy())
                    filename = out_dir / "val-imgs" / f"{name}-input-{j}" / "step-0.png"
                    Image.fromarray((normalized(input_[j]) * 255.0).cpu().detach().to(torch.uint8).permute(1, 2, 0).squeeze(-1).numpy()).save(filename)
                    filename = out_dir / "val-imgs" / f"{name}-target-{j}" / "step-0.png"
                    Image.fromarray((normalized(target[j]) * 255.0).cpu().detach().to(torch.uint8).permute(1, 2, 0).squeeze(-1).numpy()).save(filename)
                    filename = out_dir / "val-imgs" / f"{name}-prediction-{j}" / "step-0.png"
                    Image.fromarray((normalized(prediction[j]) * 255.0).cpu().detach().to(torch.uint8).permute(1, 2, 0).squeeze(-1).numpy()).save(filename)
                    tb_logger.add_image(f"val/{name}-input-{j}", normalized(input_[j]), global_step=0)
                    tb_logger.add_image(f"val/{name}-target-{j}", normalized(target[j]), global_step=0)
                    tb_logger.add_image(f"val/{name}-prediction-{j}", normalized(prediction[j]), global_step=0)
        val_loss /= len(dataloader)
        best_val_losses[name] = val_loss
        with out_dir.joinpath(f"val-loss-{name}.csv").open("w") as file:
            file.write("step, loss, time\n")
            file.write(f"0, {val_loss}, {datetime.datetime.now().isoformat()}\n")
        tb_logger.add_scalar(f"val/{name}-loss", val_loss, global_step=0)

    # Main training loop
    logger.info("Starting training loop")
    with out_dir.joinpath("train-loss.csv").open("w") as file:
        file.write("step, loss, time\n")
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
                with out_dir.joinpath("train-loss.csv").open("a") as file:
                    file.write(f"{epoch * len(train_dataloader) + batch_no}, {loss.item()}, {datetime.datetime.now().isoformat()}\n")
                tb_logger.add_scalar("train/loss", loss.item(), global_step=epoch * len(train_dataloader) + batch_no)
                batches_iter.set_postfix({"Train-Loss": loss.item()})
            batches_iter.close()

            # Validate model
            model.eval()
            all_val_loss = 0.0
            for name, dataloader in tqdm(val_dataloaders.items(), desc="Validation", unit="dataset", position=1, leave=False):
                val_loss = 0.0
                for i, sample in tqdm(enumerate(dataloader), total=len(dataloader), desc=name, unit="batch", position=2, leave=False):
                    input_ = sample["input"].to(args.device)
                    target = sample["target"].to(args.device)
                    prediction = fwd_func(input_)
                    val_loss += loss_function(prediction, target).item()
                    if i == 0:
                        for j in range(min(4, input_.shape[0])):
                            out_dir.joinpath("val-imgs", f"{name}-input-{j}").mkdir(parents=True, exist_ok=True)
                            out_dir.joinpath("val-imgs", f"{name}-target-{j}").mkdir(parents=True, exist_ok=True)
                            out_dir.joinpath("val-imgs", f"{name}-prediction-{j}").mkdir(parents=True, exist_ok=True)
                            np.save(out_dir / "val-imgs" / f"{name}-input-{j}" / f"step-{epoch + 1}.npy", input_[j].cpu().detach().numpy())
                            np.save(out_dir / "val-imgs" / f"{name}-target-{j}" / f"step-{epoch + 1}.npy", target[j].cpu().detach().numpy())
                            np.save(out_dir / "val-imgs" / f"{name}-prediction-{j}" / f"step-{epoch + 1}.npy", prediction[j].cpu().detach().numpy())
                            filename = out_dir / "val-imgs" / f"{name}-input-{j}" / f"step-{epoch + 1}.png"
                            Image.fromarray((normalized(input_[j]) * 255.0).cpu().detach().to(torch.uint8).permute(1, 2, 0).squeeze(-1).numpy()).save(filename)
                            filename = out_dir / "val-imgs" / f"{name}-target-{j}" / f"step-{epoch + 1}.png"
                            Image.fromarray((normalized(target[j]) * 255.0).cpu().detach().to(torch.uint8).permute(1, 2, 0).squeeze(-1).numpy()).save(filename)
                            filename = out_dir / "val-imgs" / f"{name}-prediction-{j}" / f"step-{epoch + 1}.png"
                            Image.fromarray((normalized(prediction[j]) * 255.0).cpu().detach().to(torch.uint8).permute(1, 2, 0).squeeze(-1).numpy()).save(filename)
                            tb_logger.add_image(f"val/{name}-input-{j}", normalized(input_[j]), global_step=epoch + 1)
                            tb_logger.add_image(f"val/{name}-target-{j}", normalized(target[j]), global_step=epoch + 1)
                            tb_logger.add_image(f"val/{name}-prediction-{j}", normalized(prediction[j]), global_step=epoch + 1)
                val_loss /= len(dataloader)
                all_val_loss += val_loss
                if val_loss < best_val_losses[name]:
                    logger.info(f"Saving new best weights on {name}")
                    logger.debug(f"    Path: {out_dir / 'weights' / f'best-{name}.pt'}")
                    best_val_losses[name] = val_loss
                    torch.save(model.state_dict(), out_dir / "weights" / f"best-{name}.pt")
                with out_dir.joinpath(f"val-loss-{name}.csv").open("a") as file:
                    file.write(f"{epoch + 1}, {val_loss}, {datetime.datetime.now().isoformat()}\n")
                tb_logger.add_scalar(f"val/{name}-loss", val_loss, global_step=epoch + 1)
                lr_scheduler.step()
                logger.info(f"Epoch {epoch + 1}: {name} val. loss = {val_loss:.3e}")
            all_val_loss /= len(val_dataloaders)
            if all_val_loss < best_val_losses["all"]:
                logger.info("Saving new best weights on all datasets")
                logger.debug(f"    Path: {out_dir / 'weights' / 'best-all.pt'}")
                best_val_losses["all"] = all_val_loss
                torch.save(model.state_dict(), out_dir / "weights" / "best-all.pt")
            with out_dir.joinpath("val-loss-all.csv").open("a") as file:
                file.write(f"{epoch + 1}, {all_val_loss}, {datetime.datetime.now().isoformat()}\n")
            tb_logger.add_scalar("val/avg-all-loss", all_val_loss, global_step=epoch + 1)
            lr_scheduler.step()
            logger.info(f"Epoch {epoch + 1}: Avg. val. loss = {all_val_loss:.3e}")

            # Save model
            if (epoch + 1) % args.model_save_freq == 0:
                logger.info("Saving weights")
                logger.debug(f"    Path: {out_dir / 'weights' / f'epoch-{epoch + 1}.pt'}")
                torch.save(model.state_dict(), out_dir / "weights" / f"epoch-{epoch + 1}.pt")
        epochs_iter.close()
    except KeyboardInterrupt:
        pass
    logger.info("Training completed")

    # Save model after training
    logger.info("Saving final weights")
    logger.debug(f"    Path: {out_dir / 'weights' / 'final.pt'}")
    torch.save(model.state_dict(), out_dir / "weights" / "final.pt")
    logger.info("Exporting weights to ONNX")
    logger.debug(f"    Path: {out_dir / 'model.onnx'}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=r"Converting a tensor to a Python boolean might cause the trace to be incorrect.*")
        torch.onnx.export(model, (torch.randn((1, *exemplary_image_shape), device=args.device),), str(out_dir / "model.onnx"))

    # Test model and log test performance
    logger.info("Testing model")
    model.load_state_dict(torch.load(out_dir / "weights" / "best-all.pt"))
    model.eval()
    with out_dir.joinpath("test-results.csv").open("w") as file:
        file.write("dataset, metric, value\n")
    out_dir.joinpath("test-imgs").mkdir(parents=True)
    all_test_loss = 0.0
    for name, dataloader in tqdm(test_dataloaders.items(), desc="Testing", unit="dataset", position=0, leave=True):
        test_loss = 0.0
        for i, sample in tqdm(enumerate(dataloader), total=len(dataloader), desc=name, unit="batch", position=1, leave=False):
            input_ = sample["input"].to(args.device)
            target = sample["target"].to(args.device)
            prediction = fwd_func(input_)
            test_loss += loss_function(prediction, target).item()
            if i == 0:
                for j in range(min(4, input_.shape[0])):
                    np.save(out_dir / "test-imgs" / f"{name}-input-{j}.npy", input_[j].cpu().detach().numpy())
                    np.save(out_dir / "test-imgs" / f"{name}-target-{j}.npy", target[j].cpu().detach().numpy())
                    np.save(out_dir / "test-imgs" / f"{name}-prediction-{j}.npy", prediction[j].cpu().detach().numpy())
                    filename = out_dir / "test-imgs" / f"{name}-input-{j}.png"
                    Image.fromarray((normalized(input_[j]) * 255.0).cpu().detach().to(torch.uint8).permute(1, 2, 0).squeeze(-1).numpy()).save(filename)
                    filename = out_dir / "test-imgs" / f"{name}-target-{j}.png"
                    Image.fromarray((normalized(target[j]) * 255.0).cpu().detach().to(torch.uint8).permute(1, 2, 0).squeeze(-1).numpy()).save(filename)
                    filename = out_dir / "test-imgs" / f"{name}-prediction-{j}.png"
                    Image.fromarray((normalized(prediction[j]) * 255.0).cpu().detach().to(torch.uint8).permute(1, 2, 0).squeeze(-1).numpy()).save(filename)
                    tb_logger.add_image(f"test/{name}-input-{j}", normalized(input_[j]), global_step=0)
                    tb_logger.add_image(f"test/{name}-target-{j}", normalized(target[j]), global_step=0)
                    tb_logger.add_image(f"test/{name}-prediction-{j}", normalized(prediction[j]), global_step=0)
        test_loss /= len(dataloader)
        all_test_loss += test_loss
        logger.info(f'Final test loss on "{name}": {test_loss:.3e}')
        with out_dir.joinpath("test-results.csv").open("a") as file:
            file.write(f"{name}, loss, {test_loss}\n")
        tb_logger.add_scalar(f"test/{name}-loss", test_loss, global_step=0)
    all_test_loss /= len(test_dataloaders)
    with out_dir.joinpath("test-results.csv").open("a") as file:
        file.write(f"all-avg, loss, {all_test_loss}\n")
    tb_logger.add_scalar("test/avg-all-loss", all_test_loss, global_step=0)
    lr_scheduler.step()
    logger.info(f"Final avg. test loss on all datasets: {all_test_loss:.3e}")

    # Cleanup
    logger.info("Cleaning up")
    tb_logger.flush()
    tb_logger.close()

    logger.info("Run completed")


if __name__ == "__main__":
    sys.exit(main())
