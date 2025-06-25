import argparse
import logging
from math import ceil
from pathlib import Path
import random
import sys
import warnings

import numpy as np
import torch
import torch.backends.cudnn

from fun.data.ct_dataset import CTPostProcessDataset
from fun.data.ellipses_dataset import EllipsesDataset
from fun.utils.formatters import ColoringIndentingFormatter


def main() -> None:
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=random.randrange(0, 2**32))
    argparser.add_argument("--precision", choices=["high", "medium", "low"], default="medium")
    argparser.add_argument("--device", type=torch.device, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    argparser.add_argument("--dataset", choices=["ellipses"], required=True)
    argparser.add_argument("--noise-level", type=float, default=0.0)
    argparser.add_argument("--angle-percent", type=float, default=0.75)
    argparser.add_argument("--num-ellipses", type=int, default=10)
    argparser.add_argument("out_dir", type=Path)
    argparser.add_argument("--smooth-data", dest="smooth", action="store_true")
    args = argparser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().handlers[0].setFormatter(ColoringIndentingFormatter(fmt="%(asctime)s.%(msecs)03d [%(levelname).4s][%(name)s]: %(message)s", datefmt="%Y.%m.%d %H:%M:%S"))
    old_showwarning = warnings.showwarning
    warnings.showwarning = lambda m, c, fn, ln, f, l: old_showwarning(m, c, fn, ln, f, l) if f is not None else logging.getLogger("py.warnings").warning(m)  # noqa: E741
    logger = logging.getLogger("fun.train")
    logger.debug("Command line arguments:")
    for arg, value in vars(args).items():
        logger.debug(f"    {arg}: {value}")

    # Configure PyTorch
    logger.info("Configuring PyTorch")
    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.float64 if args.precision == "high" else torch.float32)
    torch.set_float32_matmul_precision({"high": "highest", "medium": "high", "low": "medium"}[args.precision])

    # Setup deterministic behavior
    logger.info(f"Setting up deterministic behavior (seed: {args.seed})")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Load data
    logger.info("Creating datasets")
    match args.dataset:
        case "ellipses":
            test_dataset = EllipsesDataset(2000, 1024, args.num_ellipses, smooth = args.smooth)
            test_datasets = {
                "64x64": CTPostProcessDataset(
                    test_dataset,
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(256 * args.angle_percent)),
                    pos_count=128,
                    target_shape=(64, 64),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.device,
                ),
                "128x128": CTPostProcessDataset(
                    test_dataset,
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(512 * args.angle_percent)),
                    pos_count=256,
                    target_shape=(128, 128),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.device,
                ),
                "256x256": CTPostProcessDataset(
                    test_dataset,
                    angles=torch.linspace(0.0, torch.pi * args.angle_percent, ceil(1024 * args.angle_percent)),
                    pos_count=512,
                    target_shape=(256, 256),
                    noise_type="gaussian",
                    noise_level=args.noise_level,
                    radon_device=args.device,
                ),
            }
        case _:
            raise ValueError(f'Unknown dataset: "{args.dataset}"')

    logger.info("Saving datasets to file")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, dataset in test_datasets.items():
        dataset.save_to_file(args.out_dir / f"test-{name}.h5", progress=True)
        logger.info(f"Saved {name} dataset to file")
        logger.debug(f"    Path: {args.out_dir / f'test-{name}.h5'}")

    logger.info("Data generated successfully")


if __name__ == "__main__":
    sys.exit(main())
