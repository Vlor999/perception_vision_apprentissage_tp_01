import argparse
from src import config
from loguru import logger


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraînement optimisé pour macOS")
    parser.add_argument(
        "--cpu-only", action="store_true", help="Forcer l'utilisation du CPU uniquement"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Taille de batch personnalisée"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Nombre de workers personnalisé"
    )
    parser.add_argument(
        "--save-plots", type=str, default=True, help="Say if you save the plots or not"
    )

    args = parser.parse_args()
    # Forcer CPU si demandé
    if args.cpu_only:
        config.DEVICE = "cpu"
        config.PIN_MEMORY = False
        logger.info("**** Mode CPU forcé")

    # Override batch size si spécifié
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
        logger.info(f"**** Batch size modifiée: {config.BATCH_SIZE}")

    # Override workers si spécifié
    if args.workers is not None:
        config.NB_WORKERS = args.workers
        logger.info(f"**** Workers modifiés: {config.NB_WORKERS}")

        # Convert string to boolean properly
    if args.save_plots.lower() in ["true", "1", "yes", "on"]:
        args.save_plots = True
    elif args.save_plots.lower() in ["false", "0", "no", "off"]:
        args.save_plots = False
    else:
        raise ValueError(
            f"Invalid value for --save-plots: {args.save_plots}. Use true/false."
        )

    return args


def setup_args_prediction() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Argument for the prediction file")
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Specify a file to analyse",
        required=True,
    )
    parser.add_argument(
        "--show-all-images",
        type=str,
        default="true",
        help="Display all images or only the wrong one (true/false)",
    )

    args = parser.parse_args()

    # Convert string to boolean properly
    if args.show_all_images.lower() in ["true", "1", "yes", "on"]:
        args.show_all_images = True
    elif args.show_all_images.lower() in ["false", "0", "no", "off"]:
        args.show_all_images = False
    else:
        raise ValueError(
            f"Invalid value for --show-all-images: {args.show_all_images}. Use true/false."
        )

    return args
