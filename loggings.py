import logging
import os
from logging.handlers import RotatingFileHandler
from threading import Lock


class LoggerManager:
    _instance = None
    _lock = Lock()
    _loggers = {}

    LOG_DIR = "logs"
    LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LoggerManager, cls).__new__(cls)
                if not os.path.exists(cls.LOG_DIR):
                    os.makedirs(cls.LOG_DIR)
        return cls._instance

    def _create_logger(self, name: str, filename: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # Чтобы не дублировать в root logger

        if not logger.handlers:
            formatter = logging.Formatter(self.LOG_FORMAT)

            # File handler (всё)
            file_handler = RotatingFileHandler(
                os.path.join(self.LOG_DIR, filename),
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # Console handler (всё выше DEBUG)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    def get_main_logger(self) -> logging.Logger:
        if "main" not in self._loggers:
            self._loggers["main"] = self._create_logger("main", "main.log")
        return self._loggers["main"]

    def get_named_logger(self, name: str) -> logging.Logger:
        if name not in self._loggers:
            safe_filename = f"{name}.log".replace(" ", "_").replace(":", "_")
            self._loggers[name] = self._create_logger(name, safe_filename)
        return self._loggers[name]