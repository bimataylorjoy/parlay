"""Data contracts and validation."""

from .ingestion import ingest_csv_files
from .loaders import load_football_data_csv, load_many
from .sources import acquire_csv

__all__ = ["acquire_csv", "ingest_csv_files", "load_football_data_csv", "load_many"]
