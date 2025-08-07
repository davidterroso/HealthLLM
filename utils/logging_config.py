"""
File used to the define the configurations of logging
"""

import logging

def setup_logging(level=logging.INFO):
    """
    Function used to configure the logging presentation
    """
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=level
    )