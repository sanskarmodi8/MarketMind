from MarketMind.entity.config_entity import DataIngestionConfig
from MarketMind.constants import *
from MarketMind.utils.common import create_directories, read_yaml

# final configuration for all stages in the pipeline

class ConfigurationManager:
    def __init__(self):
        # load configs and params, and create root dir for artifacts
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARMS_FILE_PATH)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self):
        config = self.config.data_ingestion
        params = self.params.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            asset=params.asset,
            start_date=params.start_date,
            end_date=params.end_date
        )