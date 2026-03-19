# src/utils.py
import logging


def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger that writes to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def parse_block_id(block_id: str) -> tuple:
    """Parse a Block_ID string 'lat_lon' into (lat, lon) floats."""
    parts = block_id.split('_')
    if len(parts) != 2:
        raise ValueError(f"Unexpected Block_ID format: '{block_id}'")
    return float(parts[0]), float(parts[1])
