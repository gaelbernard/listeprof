import sys
from dspace_rest_client.client import DSpaceClient
from datetime import datetime
import time
from abc import abstractmethod
from pathlib import Path
import logging

class Logger:

    def __init__(self, path='logs'):
        self.path = path
        self.logger = self.initialize()

    def initialize(self):
        LOG_DIR = Path("logs")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f"process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode="w", encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
            force=True,
        )
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logger = logging.getLogger("pipeline")

        def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

        sys.excepthook = log_uncaught_exceptions
        return logger


