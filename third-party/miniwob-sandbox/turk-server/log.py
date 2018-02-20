import logging
import datetime


def create_logger(name, file_path, level='DEBUG'):
    """Create a logger with the specified name."""
    logger = logging.getLogger(name)

    if isinstance(level, str):
        level = logging._levelNames[level]
    logger.setLevel(level)

    log_formatter = logging.Formatter("%(asctime)s [%(name)-6.6s] [%(levelname)-5.5s] %(message)s")
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    return logger


# epoch time to pretty printed string
epoch_to_str = lambda t: datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]