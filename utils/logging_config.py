"""
File used to the define the configurations of logging
"""

import logging
import sys
from tqdm import tqdm

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

class TqdmLogger:
    """
    Wrapper around the tqdm write function to 
    output it without ruining tqdm progress bar
    """
    @staticmethod
    def info(msg: str) -> None:
        """
        Displays the message as the level info

        Args:
            msg (str): message desired to display

        Returns:
            None
        """
        tqdm.write(f"[INFO] {msg}")
    @staticmethod
    def warning(msg):
        """
        Displays the message as the level warning

        Args:
            msg (str): message desired to display

        Returns:
            None
        """
        tqdm.write(f"[WARNING] {msg}")
    @staticmethod
    def error(msg):
        """
        Displays the message as the level error

        Args:
            msg (str): message desired to display

        Returns:
            None
        """
        tqdm.write(f"[ERROR] {msg}")
