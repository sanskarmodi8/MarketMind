from pathlib import Path
from dataclasses import dataclass

# config entities for each stage in pipeline

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_path: Path
    asset: str
    start_date: str
    end_date: str

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    data_path: Path
    preprocessed_data_path: Path