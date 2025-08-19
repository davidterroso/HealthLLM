"""
File used to the define the configurations of logging
"""

import logging
import sys

def setup_logging(level=logging.WARNING):
    """
    Function used to configure the logging presentation

    Args:
        level (int): minimum level that leads to logging

    Returns:
        None
    """
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=level,
        stream=sys.stdout
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
