import argparse
from src import config
from loguru import logger


def setup_args():
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

    args = parser.parse_args()
    # Forcer CPU si demandé
    if args.cpu_only:
        config.DEVICE = "cpu"
        config.PIN_MEMORY = False
        logger.debug("**** Mode CPU forcé")

    # Override batch size si spécifié
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
        logger.debug(f"**** Batch size modifiée: {config.BATCH_SIZE}")

    # Override workers si spécifié
    if args.workers is not None:
        config.NB_WORKERS = args.workers
        logger.debug(f"**** Workers modifiés: {config.NB_WORKERS}")
