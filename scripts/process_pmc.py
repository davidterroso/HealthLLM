"""
Script called to process the data using GitHub Actions
"""

from prepare_data.get_data import data_pipeline
from utils.load_config import load_config

config = load_config()

if __name__ == "__main__":
    data_pipeline(collection_name=config["collection_name"],
                  tar_file_dir=config["tar_file_dir"])
