"""
Wrapper to safely get the configurations file, handling the 
errors gracefully
"""

import os
import json
import logging

def load_config() -> dict:
    """
    Function that loads the JSON file with the 
    configuration and handles the errors gracefully

    Args:
        None

    Returns:
        (dict): JSON file converted to a dictionary
    """
    config_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'data_handling', 'config.json'
    )
    try:
        with open(os.path.abspath(config_path), "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning("Config file not found. Using defaults.")
        return {}
    except json.JSONDecodeError as e:
        logging.error("Invalid JSON in config file: %s", e)
        return {}
