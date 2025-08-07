"""
Script called to process the data using GitHub Actions
"""

from data_handling.get_data import data_pipeline

data_pipeline(collection_name="pmc_embeddings")
