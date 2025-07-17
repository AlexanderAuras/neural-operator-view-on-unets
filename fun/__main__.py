# import os
# import sys
#
#
# sys.path.append(os.path.join(os.getcwd(), ".."))
#
# for desired_path in ("/software/gcc/10.5/bin", "/usr/local/cuda-12.1/bin"):
#     print(desired_path, end="")
#     if desired_path not in os.environ["PATH"]:
#         print(" - adding")
#         os.environ["PATH"] = f"{desired_path}:{os.environ['PATH']}"
#     else:
#         print(" - already added")
# print(os.environ["PATH"])


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

from cno.CNOModule import CNO
from neuralop.models import UNO
import numpy as np
import PIL.Image as Image
import randomname
import torch
from torch import Tensor, nn
import torch.backends.cudnn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.tensorboard
import torchinfo
import torchmetrics.functional.image
import torchview
from tqdm.auto import tqdm, trange

from fun.data.ct_dataset import CTPostProcessDataset
from fun.data.ellipses_dataset import EllipsesDataset
from fun.data.multi_res_batch_sampler import MultiResolutionBatchSampler
from fun.models.custom_unet import CustomUNet
from fun.models.dncnn import DnCNN
from fun.models.interp_unet import InterpolatingUNet
from fun.models.spectral_unet import SpectralResUNet
from fun.utils.diff_utils import DiffConv2d


BASE_OUT_DIR = Path(__file__).resolve().parents[1] / "runs"
BASE_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

LOG_VAL_WEIGHTS = False
LOG_TRAIN_WEIGHTS_GRADS = False


def setup_logging(logging_config_path: str | Path, out_dir: str | Path | None = None) -> None:
    try:
        logging.config.fileConfig(Path(logging_config_path).resolve())
        if out_dir is not None:
            for logger in [logging.getLogger(name) for name in [*logging.root.manager.loggerDict.keys(), None]]:
                for handler in logger.handlers:
                    if isinstance(handler, logging.FileHandler):
                        handler.baseFilename = str(Path(out_dir) / Path(handler.baseFilename).name)
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
    global BASE_DATA_DIR
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--seed", type=int, default=random.randrange(0, 2**32))
    argparser.add_argument("--devices", type=torch.device, nargs="+", default=[torch.device("cuda" if torch.cuda.is_available() else "cpu")])
    argparser.add_argument("--num-workers", type=int, default=os.cpu_count())
    argparser.add_argument("--no-compile", dest="compile", action="store_false")
    argparser.add_argument("--data-parallel", action="store_true")
    argparser.add_argument("--use-checkpointing", action="store_true")
    argparser.add_argument("--forced-run-name", type=str, default=None)
    argparser.add_argument("--precision", choices=["high", "medium", "low"], default="medium")
    argparser.add_argument("--dataset", choices=["ellipses-64x64", "ellipses-128x128", "ellipses-256x256", "ellipses-mixed", "ellipses-sweep"], required=True)
    argparser.add_argument("--model", choices=["unet", "dncnn", "unet-interp", "specResU", "spatResU", "specRes", "smallResU", "unet-custom", "cno", "uno", "diff"], required=True)
    argparser.add_argument("--kbase", type=int, default=256)
    argparser.add_argument("--weights", type=Path, default=None)
    argparser.add_argument("--test-only", action="store_true")
    argparser.add_argument("--batch-size", type=int, default=32)
    argparser.add_argument("--accumulation-steps", type=int, default=1)
    argparser.add_argument("--max-epochs", type=int, default=10)
    argparser.add_argument("--lr", type=float, default=1e-3)
    argparser.add_argument("--model-save-freq", type=int, default=1)
    argparser.add_argument("--noise-level", type=float, default=0.0)
    argparser.add_argument("--angle-percent", type=float, default=0.75)
    argparser.add_argument("--num-ellipses", type=int, default=10)
    argparser.add_argument("--smooth-data", dest="smooth", action="store_true")
    argparser.add_argument("--interp-mode", action="store_true")
    argparser.add_argument("--unet-convs", type=int, default=1)
    argparser.add_argument("--pooling-base-size", type=int, default=None)
    argparser.add_argument("--cpu-opt-state", action="store_true")
    argparser.add_argument("--resize-input-size", type=int, default=None)
    args = argparser.parse_args()

    if args.forced_run_name is not None:
        out_dir = BASE_OUT_DIR / args.forced_run_name
    else:
        out_dir = BASE_OUT_DIR / randomname.get_name()
    if args.debug:
        out_dir = out_dir.parent.joinpath("_debug")
        if out_dir.exists():
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    if not os.isatty(sys.stdout.fileno()):
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
            logger.debug(f"    Adding {file}")
            code_archive.write(file, file.relative_to(Path(__file__).parents[1].resolve()))

    # Validate cli arguments
    if args.debug:
        logger.warning("Executing debug run")
    if args.dataset == "ellipses-sweep" and not args.test_only:
        logger.warning("Using ellipses-sweep dataset, skipping training")
        args.test_only = True
    if args.test_only:
        logger.warning("Test only run, skipping training")
        if args.weights is None:
            logger.error("No weights provided for test only run")
            exit(-1)
    if "cuda" in {x.type for x in args.devices} and not torch.cuda.is_available():
        logger.warning("No CUDA devices available, falling back to CPU")
        args.devices = [torch.device("cpu") if x.type == "cuda" else x for x in args.devices]

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
    if args.smooth:
        BASE_DATA_DIR = BASE_DATA_DIR / "smooth"  # pyright: ignore [reportConstantRedefinition]
    in_size = None
    match args.dataset:
        case "ellipses-64x64":
            in_size = 64
            train_dataset = CTPostProcessDataset(
                EllipsesDataset.from_file(BASE_DATA_DIR / "train.h5"),
                angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                pos_count=128,
                target_shape=(64, 64),
                noise_type="gaussian",
                noise_level=args.noise_level,
                radon_device=args.devices[0],
            )
            train_batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(train_dataset), batch_size=args.batch_size, drop_last=False)
            val_datasets = {
                "64x64": CTPostProcessDataset(
                    EllipsesDataset.from_file(BASE_DATA_DIR / "val.h5"),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                    pos_count=128,
                    target_shape=(64, 64),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
            }
            test_datasets = {
                "64x64": CTPostProcessDataset(
                    EllipsesDataset.from_file(BASE_DATA_DIR / "test.h5"),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                    pos_count=128,
                    target_shape=(64, 64),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
                "128x128": CTPostProcessDataset(
                    EllipsesDataset.from_file(BASE_DATA_DIR / "test.h5"),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                    pos_count=256,
                    target_shape=(128, 128),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
                "256x256": CTPostProcessDataset(
                    EllipsesDataset.from_file(BASE_DATA_DIR / "test.h5"),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                    pos_count=512,
                    target_shape=(256, 256),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
            }
            exemplary_image_shape = (1, 64, 64)
        case "ellipses-128x128":
            in_size = 128
            train_dataset = CTPostProcessDataset(
                EllipsesDataset.from_file(BASE_DATA_DIR / "train.h5"),
                angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                pos_count=256,
                target_shape=(128, 128),
                noise_type="gaussian",
                noise_level=args.noise_level,
                radon_device=args.devices[0],
            )
            train_batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(train_dataset), batch_size=args.batch_size, drop_last=False)
            val_datasets = {
                "128x128": CTPostProcessDataset(
                    EllipsesDataset.from_file(BASE_DATA_DIR / "val.h5"),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                    pos_count=256,
                    target_shape=(128, 128),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
            }
            test_datasets = {
                "64x64": CTPostProcessDataset(
                    EllipsesDataset.from_file(BASE_DATA_DIR / "test.h5"),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                    pos_count=128,
                    target_shape=(64, 64),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
                "128x128": CTPostProcessDataset(
                    EllipsesDataset.from_file(BASE_DATA_DIR / "test.h5"),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                    pos_count=256,
                    target_shape=(128, 128),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
                "256x256": CTPostProcessDataset(
                    EllipsesDataset.from_file(BASE_DATA_DIR / "test.h5"),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                    pos_count=512,
                    target_shape=(256, 256),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
            }
            exemplary_image_shape = (1, 128, 128)
        case "ellipses-256x256":
            in_size = 256
            train_dataset = CTPostProcessDataset(
                EllipsesDataset.from_file(BASE_DATA_DIR / "train.h5"),
                angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                pos_count=512,
                target_shape=(256, 256),
                noise_type="gaussian",
                noise_level=args.noise_level,
                radon_device=args.devices[0],
            )
            train_batch_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(train_dataset), batch_size=args.batch_size, drop_last=False)
            val_datasets = {
                "256x256": CTPostProcessDataset(
                    EllipsesDataset.from_file(BASE_DATA_DIR / "val.h5"),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                    pos_count=512,
                    target_shape=(256, 256),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
            }
            test_datasets = {
                "64x64": CTPostProcessDataset(
                    EllipsesDataset.from_file(BASE_DATA_DIR / "test.h5"),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                    pos_count=128,
                    target_shape=(64, 64),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
                "128x128": CTPostProcessDataset(
                    EllipsesDataset.from_file(BASE_DATA_DIR / "test.h5"),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                    pos_count=256,
                    target_shape=(128, 128),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
                "256x256": CTPostProcessDataset(
                    EllipsesDataset.from_file(BASE_DATA_DIR / "test.h5"),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                    pos_count=512,
                    target_shape=(256, 256),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
            }
            exemplary_image_shape = (1, 256, 256)
        case "ellipses-mixed":
            if args.model == "unet-interp":
                raise ValueError("Model 'cno' is not supported for datasets with variable resolutions")
            train_datasets = [
                CTPostProcessDataset(
                    torch.utils.data.Subset(EllipsesDataset.from_file(BASE_DATA_DIR / "train.h5"), range(2133)),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                    pos_count=128,
                    target_shape=(64, 64),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
                CTPostProcessDataset(
                    torch.utils.data.Subset(EllipsesDataset.from_file(BASE_DATA_DIR / "train.h5"), range(2133)),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                    pos_count=256,
                    target_shape=(128, 128),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
                CTPostProcessDataset(
                    torch.utils.data.Subset(EllipsesDataset.from_file(BASE_DATA_DIR / "train.h5"), range(2133)),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                    pos_count=512,
                    target_shape=(256, 256),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
            ]
            train_dataset = torch.utils.data.ConcatDataset(train_datasets)
            train_batch_sampler = MultiResolutionBatchSampler([len(x) for x in train_datasets], batch_size=args.batch_size, shuffle=True, drop_incomplete=False)
            val_datasets = {
                "64x64": CTPostProcessDataset(
                    torch.utils.data.Subset(EllipsesDataset.from_file(BASE_DATA_DIR / "val.h5"), range(533)),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                    pos_count=128,
                    target_shape=(64, 64),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
                "128x128": CTPostProcessDataset(
                    torch.utils.data.Subset(EllipsesDataset.from_file(BASE_DATA_DIR / "val.h5"), range(533)),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                    pos_count=256,
                    target_shape=(128, 128),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
                "256x256": CTPostProcessDataset(
                    torch.utils.data.Subset(EllipsesDataset.from_file(BASE_DATA_DIR / "val.h5"), range(533)),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                    pos_count=512,
                    target_shape=(256, 256),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
            }
            test_datasets = {
                "64x64": CTPostProcessDataset(
                    EllipsesDataset.from_file(BASE_DATA_DIR / "test.h5"),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                    pos_count=128,
                    target_shape=(64, 64),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
                "128x128": CTPostProcessDataset(
                    EllipsesDataset.from_file(BASE_DATA_DIR / "test.h5"),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                    pos_count=256,
                    target_shape=(128, 128),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
                "256x256": CTPostProcessDataset(
                    EllipsesDataset.from_file(BASE_DATA_DIR / "test.h5"),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                    pos_count=512,
                    target_shape=(256, 256),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.devices[0],
                ),
            }
            exemplary_image_shape = (1, 256, 256)
        case "ellipses-sweep":
            if args.model == "cno" and args.resize_input_size is None:
                raise ValueError("Model 'cno' is not supported for datasets with variable resolutions")
            train_dataset = cast(torch.utils.data.Dataset[dict[str, Tensor]], [{"input": torch.empty(0), "target": torch.empty(0)}])
            train_batch_sampler = MultiResolutionBatchSampler([1], batch_size=args.batch_size, shuffle=True, drop_incomplete=False)
            val_datasets = {}
            test_datasets = {
                f"{r}x{r}": CTPostProcessDataset(
                    torch.utils.data.Subset(EllipsesDataset.from_file(BASE_DATA_DIR / "test.h5"), range(200)),
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(4 * r * args.angle_percent)),
                    pos_count=2 * r,
                    target_shape=(r, r),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                )
                #for r in range(*((64, 1025, 64) if args.model == "unet-interp" else (16, 1025, 16)))
                for r in ([x for i in range(4, 9) for x in [2**i, 2**i+2**(i-2)+1, 2**i+2**(i-1), 2**(i+1)-2**(i-2)-1]] + [2**9])
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
        case "unet":
            model = CustomUNet(1, 1, use_checkpointing=args.use_checkpointing, nonresize_convs_per_block=args.unet_convs)
        case "diff":
            model = CustomUNet(1, 1, use_checkpointing=args.use_checkpointing, nonresize_convs_per_block=args.unet_convs, conv_type=DiffConv2d, conv_kwargs={})
        case "dncnn":
            model = DnCNN(1, use_checkpointing=args.use_checkpointing)
        case "unet-interp":
            model = InterpolatingUNet(1, 1, base_input_size=64, max_scale_factor=4, use_checkpointing=args.use_checkpointing)
        case "specResU":
            model = SpectralResUNet(1, 1, parametrization="spectral", u_shape=True, kbase1=args.kbase, kbase2=args.kbase)
        case "specRes":
            model = SpectralResUNet(1, 1, parametrization="spectral", u_shape=False, kbase1=args.kbase, kbase2=args.kbase)
        case "spatResU":
            model = SpectralResUNet(1, 1, parametrization="spatial", u_shape=True, kbase1=args.kbase, kbase2=args.kbase)
        case "cno":
            model = CNO(in_dim=1, in_size=in_size, N_layers=4, N_res=3, N_res_neck=3, channel_multiplier=128)
        case "unet-custom":
            model = CustomUNet(1, 1, use_checkpointing=args.use_checkpointing, nonresize_convs_per_block=args.unet_convs, optional_pool_base_size=args.pooling_base_size)
        case "uno":
            model = UNO(
                1,
                1,
                64,
                n_layers=9,
                uno_out_channels=[64, 64, 64, 64, 64, 64, 64, 64, 64],
                uno_n_modes=[[256, 256], [128, 128], [64, 64], [32, 32], [16, 16], [32, 32], [64, 64], [128, 128], [256, 256]],
                uno_scalings=[[1.0, 1.0], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
                skip="linear",
                fno_skip="linear",
                channel_mlp_skip="linear",
            )
        case _:
            raise ValueError(f'Unknown model: "{args.model}"')
    logger.info("Rendering compute graph")
    compute_graph = torchview.draw_graph(model, input_size=(1, *exemplary_image_shape), show_shapes=True, device="cpu")
    compute_graph.visual_graph.render("compute-graph", directory=out_dir, format="pdf")
    out_dir.joinpath("compute-graph").unlink()
    logger.info("Calculating model summary")
    model_statistics = torchinfo.summary(model, input_size=(1, *exemplary_image_shape), verbose=0, device="cpu")
    logger.info(model_statistics)
    out_dir.joinpath("weights").mkdir(parents=True)
    if args.weights is not None:
        logger.info(f"Loading initial weights {args.weights}")
        model = model.cpu()
        weights = torch.load(args.weights, map_location="cpu")
        for k, v in list(weights.items()):
            if k.startswith("_InterpolatingUNet_"):
                weights[k[19:]] = v
                del weights[k]
        model.load_state_dict(weights)
    else:
        logger.info("Saving initial weights")
        logger.debug(f"    Path: {out_dir / 'weights' / 'initial.pt'}")
        [logger.debug(f"    Path: {out_dir / 'weights' / f'best-{name}'}") for name in val_dataloaders.keys()]
        logger.debug(f"    Path: {out_dir / 'weights' / 'best-all'}")
        torch.save(model.state_dict(), out_dir / "weights" / "initial.pt")
        [torch.save(model.state_dict(), out_dir / "weights" / f"best-{name}.pt") for name in val_dataloaders.keys()]
        torch.save(model.state_dict(), out_dir / "weights" / "best-all.pt")
    if hasattr(model, "to_multi_dev"):
        model = model.to_multi_dev(*args.devices)
    else:
        model = model.to(args.devices[0])
    if args.data_parallel:
        logger.info("Parallelizing model")
        model = cast(nn.Module, nn.DataParallel(model))
    if args.compile:
        logger.info("Compiling model")
        fwd_func = torch.compile(model)
    else:
        fwd_func = model.__call__

    # Create loss function, optimizer, and learning rate scheduler
    logger.info("Creating loss function, optimizer, and learning rate scheduler")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.MSELoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    # Setting up model, metrics, data, etc. logging
    logger.info("Creating tensorboard logger")
    tb_logger = torch.utils.tensorboard.writer.SummaryWriter(out_dir)
    best_val_losses = {**{name: float("inf") for name in val_dataloaders.keys()}, "all": float("inf")}

    # Initial validation before training
    if not args.test_only:
        model.eval()
        logger.info("Performing initial validation")
        for name, dataloader in tqdm(val_dataloaders.items(), desc="Validation", unit="dataset", position=0, leave=False):
            val_loss = 0.0
            for i, sample in tqdm(enumerate(dataloader), total=len(dataloader), desc=name, unit="batch", position=1, leave=False):
                input_ = sample["input"].to(args.devices[0])
                target = sample["target"].to(args.devices[0])
                if args.resize_input_size is not None:
                    input_ = nn.functional.interpolate(input_, size=(args.resize_input_size, args.resize_input_size), mode="bilinear", align_corners=False)
                    target = nn.functional.interpolate(target, size=(args.resize_input_size, args.resize_input_size), mode="bilinear", align_corners=False)
                prediction = fwd_func(input_)
                val_loss += loss_function(prediction, target).item() * ((input_.shape[-1] / 100) if args.interp_mode else 1)
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
                file.write("step,loss,time\n")
                file.write(f"0,{val_loss},{datetime.datetime.now().isoformat()}\n")
            tb_logger.add_scalar(f"val/{name}-loss", val_loss, global_step=0)
        if LOG_VAL_WEIGHTS:
            for name, parameter in model.named_parameters():
                tb_logger.add_histogram("val/" + name, parameter.flatten(), global_step=0)

        # Main training loop
        logger.info("Starting training loop")
        with out_dir.joinpath("train-loss.csv").open("w") as file:
            file.write("step,loss,time\n")
        try:
            epoch = 0
            epochs_iter = trange(args.max_epochs, desc="Epochs", unit="epoch", position=0, leave=False)
            for epoch in epochs_iter:
                # Train one epoch
                model.train()
                batches_iter = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training", unit="batch", position=1, leave=False)
                optimizer.zero_grad()
                for batch_no, sample in batches_iter:
                    input_ = sample["input"].to(args.devices[0])
                    target = sample["target"].to(args.devices[0])
                    if args.resize_input_size is not None:
                        input_ = nn.functional.interpolate(input_, size=(args.resize_input_size, args.resize_input_size), mode="bilinear", align_corners=False)
                        target = nn.functional.interpolate(target, size=(args.resize_input_size, args.resize_input_size), mode="bilinear", align_corners=False)
                    with torch.enable_grad():
                        prediction = fwd_func(input_)
                        loss = loss_function(prediction, target) * ((input_.shape[-1] / 100) if args.interp_mode else 1) / args.accumulation_steps
                        loss.backward()
                    if args.interp_mode:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # pyright: ignore [reportPrivateImportUsage]
                    if (batch_no + 1) % args.accumulation_steps == 0 or batch_no == len(train_dataloader) - 1:
                        if args.cpu_opt_state:
                            model.to("cpu")
                        optimizer.step()
                        if LOG_TRAIN_WEIGHTS_GRADS:
                            for name, parameter in model.named_parameters():
                                if not (parameter.isnan().any() or parameter.isinf().any()) and parameter.numel() > 1:
                                    tb_logger.add_histogram("train/" + name, parameter.flatten(), global_step=epoch * len(train_dataloader) + batch_no, bins="auto")
                            for name, parameter in model.named_parameters():
                                if parameter.grad is not None and not (parameter.grad.isnan().any() or parameter.grad.isinf().any()) and parameter.grad.numel() > 1:
                                    tb_logger.add_histogram("train/" + name + ".grad", parameter.grad.flatten(), global_step=epoch * len(train_dataloader) + batch_no, bins="auto")
                        optimizer.zero_grad()
                        if args.cpu_opt_state:
                            model.to(args.devices[0])
                    with out_dir.joinpath("train-loss.csv").open("a") as file:
                        file.write(f"{epoch * len(train_dataloader) + batch_no},{loss.item()},{datetime.datetime.now().isoformat()}\n")
                    tb_logger.add_scalar("train/loss", loss.item(), global_step=epoch * len(train_dataloader) + batch_no)
                    batches_iter.set_postfix({"Train-Loss": loss.item()})
                batches_iter.close()

                # Validate model
                model.eval()
                all_val_loss = 0.0
                for name, dataloader in tqdm(val_dataloaders.items(), desc="Validation", unit="dataset", position=1, leave=False):
                    val_loss = 0.0
                    for i, sample in tqdm(enumerate(dataloader), total=len(dataloader), desc=name, unit="batch", position=2, leave=False):
                        input_ = sample["input"].to(args.devices[0])
                        target = sample["target"].to(args.devices[0])
                        if args.resize_input_size is not None:
                            input_ = nn.functional.interpolate(input_, size=(args.resize_input_size, args.resize_input_size), mode="bilinear", align_corners=False)
                            target = nn.functional.interpolate(target, size=(args.resize_input_size, args.resize_input_size), mode="bilinear", align_corners=False)
                        prediction = fwd_func(input_)
                        val_loss += loss_function(prediction, target).item() * ((input_.shape[-1] / 100) if args.interp_mode else 1)
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
                        file.write(f"{epoch + 1},{val_loss},{datetime.datetime.now().isoformat()}\n")
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
                    file.write(f"{epoch + 1},{all_val_loss},{datetime.datetime.now().isoformat()}\n")
                tb_logger.add_scalar("val/avg-all-loss", all_val_loss, global_step=epoch + 1)
                lr_scheduler.step()
                logger.info(f"Epoch {epoch + 1}: Avg. val. loss = {all_val_loss:.3e}")
                if LOG_VAL_WEIGHTS:
                    for name, parameter in model.named_parameters():
                        tb_logger.add_histogram("val/" + name, parameter.flatten(), global_step=epoch + 1)

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
        if args.model not in ["specResU", "specRes", "spatResU", "cno", "uno"]:
            logger.info("Exporting weights to ONNX")
            logger.debug(f"    Path: {out_dir / 'model.onnx'}")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=r"Converting a tensor to a Python boolean might cause the trace to be incorrect.*")
                torch.onnx.export(model.cpu(), (torch.randn((1, *exemplary_image_shape), device="cpu"),), str(out_dir / "model.onnx"))

    # Test model and log test performance
    logger.info("Testing model")
    if not args.test_only:
        model.load_state_dict(torch.load(out_dir / "weights" / "best-all.pt"))
    model.eval()
    with out_dir.joinpath("test-results.csv").open("w") as file:
        file.write("dataset,metric,value\n")
    with out_dir.joinpath("baseline.csv").open("w") as file:
        file.write("dataset,metric,value\n")
    out_dir.joinpath("test-imgs").mkdir(parents=True)
    all_test_loss = 0.0
    all_baseline_loss = 0.0
    all_test_mse = 0.0
    all_baseline_mse = 0.0
    all_test_psnr = 0.0
    all_baseline_psnr = 0.0
    all_test_ssim = 0.0
    all_baseline_ssim = 0.0
    for name, dataloader in tqdm(test_dataloaders.items(), desc="Testing", unit="dataset", position=0, leave=True):
        try:
            test_loss = 0.0
            baseline_loss = 0.0
            test_mse = 0.0
            baseline_mse = 0.0
            test_psnr = 0.0
            baseline_psnr = 0.0
            test_ssim = 0.0
            baseline_ssim = 0.0
            for i, sample in tqdm(enumerate(dataloader), total=len(dataloader), desc=name, unit="batch", position=1, leave=False):
                input_ = sample["input"].to(args.devices[0])
                target = sample["target"].to(args.devices[0])
                if args.resize_input_size is not None:
                    input_ = nn.functional.interpolate(input_, size=(args.resize_input_size, args.resize_input_size), mode="bilinear", align_corners=False)
                    target = nn.functional.interpolate(target, size=(args.resize_input_size, args.resize_input_size), mode="bilinear", align_corners=False)
                prediction = fwd_func(input_)
                test_loss += loss_function(prediction, target).item() * ((input_.shape[-1] / 100) if args.interp_mode else 1)
                baseline_loss += loss_function(input_, target).item() * ((input_.shape[-1] / 100) if args.interp_mode else 1)
                test_mse += torch.nn.functional.mse_loss(prediction, target).item()
                baseline_mse += torch.nn.functional.mse_loss(input_, target).item()
                test_psnr += torchmetrics.functional.image.peak_signal_noise_ratio(prediction, target, data_range=1.0, dim=0).item()
                baseline_psnr += torchmetrics.functional.image.peak_signal_noise_ratio(input_, target, data_range=1.0, dim=0).item()
                test_ssim += cast(Tensor, torchmetrics.functional.image.structural_similarity_index_measure(prediction, target, data_range=1.0)).item()
                baseline_ssim += cast(Tensor, torchmetrics.functional.image.structural_similarity_index_measure(input_, target, data_range=1.0)).item()
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
            baseline_loss /= len(dataloader)
            all_baseline_loss += baseline_loss
            all_test_loss += test_loss
            all_baseline_mse += baseline_mse
            all_test_mse += test_mse
            all_baseline_psnr += baseline_psnr
            all_test_psnr += test_psnr
            all_baseline_ssim += baseline_ssim
            all_test_ssim += test_ssim
            logger.info(f'Final test loss on "{name}": {test_loss:.3e}')
            with out_dir.joinpath("test-results.csv").open("a") as file:
                file.write(f"{name},loss,{test_loss}\n")
                file.write(f"{name},mse,{test_mse}\n")
                file.write(f"{name},psnr,{test_psnr}\n")
                file.write(f"{name},ssim,{test_ssim}\n")
            with out_dir.joinpath("baseline.csv").open("a") as file:
                file.write(f"{name},loss,{baseline_loss}\n")
                file.write(f"{name},mse,{baseline_mse}\n")
                file.write(f"{name},psnr,{baseline_psnr}\n")
                file.write(f"{name},ssim,{baseline_ssim}\n")
            tb_logger.add_scalar(f"test/{name}-loss", test_loss, global_step=0)
            tb_logger.add_scalar(f"test/{name}-mse", test_mse, global_step=0)
            tb_logger.add_scalar(f"test/{name}-psnr", test_psnr, global_step=0)
            tb_logger.add_scalar(f"test/{name}-ssim", test_ssim, global_step=0)
            tb_logger.add_scalar(f"baseline/{name}-loss", baseline_loss, global_step=0)
            tb_logger.add_scalar(f"baseline/{name}-mse", baseline_mse, global_step=0)
            tb_logger.add_scalar(f"baseline/{name}-psnr", baseline_psnr, global_step=0)
            tb_logger.add_scalar(f"baseline/{name}-ssim", baseline_ssim, global_step=0)
        except Exception as e:
            logger.error(f"Error during testing on dataset {name}: {e}")
            continue
    all_test_loss /= len(test_dataloaders)
    with out_dir.joinpath("test-results.csv").open("a") as file:
        file.write(f"all-avg,loss,{all_test_loss}\n")
        file.write(f"all-avg,mse,{all_test_mse}\n")
        file.write(f"all-avg,psnr,{all_test_psnr}\n")
        file.write(f"all-avg,ssim,{all_test_ssim}\n")
    with out_dir.joinpath("baseline.csv").open("a") as file:
        file.write(f"all-avg,loss,{all_baseline_loss}\n")
        file.write(f"all-avg,mse,{all_baseline_mse}\n")
        file.write(f"all-avg,psnr,{all_baseline_psnr}\n")
        file.write(f"all-avg,ssim,{all_baseline_ssim}\n")
    tb_logger.add_scalar("test/avg-all-loss", all_test_loss, global_step=0)
    tb_logger.add_scalar("test/avg-all-mse", all_test_mse, global_step=0)
    tb_logger.add_scalar("test/avg-all-psnr", all_test_psnr, global_step=0)
    tb_logger.add_scalar("test/avg-all-ssim", all_test_ssim, global_step=0)
    tb_logger.add_scalar("baseline/avg-all-loss", all_baseline_loss, global_step=0)
    tb_logger.add_scalar("baseline/avg-all-mse", all_baseline_mse, global_step=0)
    tb_logger.add_scalar("baseline/avg-all-psnr", all_baseline_psnr, global_step=0)
    tb_logger.add_scalar("baseline/avg-all-ssim", all_baseline_ssim, global_step=0)
    logger.info(f"Final avg. test loss on all datasets: {all_test_loss:.3e}")

    # Cleanup
    logger.info("Cleaning up")
    tb_logger.flush()
    tb_logger.close()

    logger.info("Run completed")


if __name__ == "__main__":
    sys.exit(main())
