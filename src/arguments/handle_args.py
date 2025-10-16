import argparse
from src import config
from loguru import logger


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Setup argument")

    parser.add_argument(
        "--cpu-only", action="store_true", help="Force the usage of CPU over any GPU"
    )
    parser.add_argument("--epoch-size", type=int, default=None, help="Number of epoch")
    parser.add_argument("--batch-size", type=int, default=None, help="Number of batch")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    parser.add_argument(
        "--save-plots",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Display the plots (true/false)",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Save the current model (true/false)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simple",
        choices=["simple", "deeper", "vgg_inspired", "resnet", "resnet_unfrozen"],
        help="Choose the model architecture: simple (SimpleDetector), deeper (DeeperDetector), vgg_inspired (VGGInspired), resnet (ResnetObjectDetector frozen), or resnet_unfrozen (ResnetObjectDetector unfrozen)",
    )

    args = parser.parse_args()
    # Using CPU if asked
    if args.cpu_only:
        config.DEVICE = "cpu"
        config.PIN_MEMORY = False
        logger.info("**** Mode CPU forcé")

    # Override the epoch size
    if args.epoch_size:
        config.NUM_EPOCHS = args.epoch_size
        logger.info(f"**** Batch size modifiée: {config.NUM_EPOCHS}")

    # Override batch size si spécifié
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
        logger.info(f"**** Batch size modifiée: {config.BATCH_SIZE}")

    # Override workers si spécifié
    if args.workers is not None:
        config.NB_WORKERS = args.workers
        logger.info(f"**** Workers modifiés: {config.NB_WORKERS}")

    # Convert string to boolean properly
    args.save_plots = True if args.save_plots == "true" else False
    args.save_model = True if args.save_model == "true" else False

    return args


def setup_args_prediction() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Argument for the prediction file")

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--directory",
        type=str,
        default=None,
        help="Specify a file to analyse",
    )
    group.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Selected a specific file to test for type and bounding box"
    )
    parser.add_argument(
        "--show-all-images",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Display all images or only the wrong one (true/false)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["best", "last"],
        default="best",
        help="choose between the last model saved and the best model"
    )

    parser.add_argument(
        "--save-file",
        type=str,
        default="false",
        choices=['true', 'false'],
        help="If you are using one test file you can save the file or not"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="You must give an output filename if you want to save the file"
    )

    args = parser.parse_args()

    args.show_all_images = True if args.show_all_images == "true" else "false"
    args.model = config.BEST_MODEL_PATH if args.model == "best" else config.LAST_MODEL_PATH
    args.save_file = True if args.save_file == 'true' else False

    return args
