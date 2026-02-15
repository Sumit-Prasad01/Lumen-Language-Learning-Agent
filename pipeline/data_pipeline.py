from src.data_ingestion import DataIngestion
from src.data_processor import DataProcessor
from config.paths_config import *
from config.models_list import SPACY_MODELS
from utils.logger import get_logger

logger = get_logger(__name__)


class DataPipeline:

    def __init__(self):
        pass

    def run_data_pipeline(self):

        STAGE_NAME = "Data Ingestion"

        ingestion = DataIngestion(REPO_URL, REPO_DIR, OUTPUT_DIR)
        ingestion.run()

        logger.info(f"{STAGE_NAME} completed successfully.")

        STAGE_NAME = "Data Processing"

        processor = DataProcessor(SPACY_MODELS, RAW_WORD_LIST_DIR)
        processor.process_all_languages()

        logger.info(f"{STAGE_NAME} completed successfully.")

    

if __name__ == "__main__":

    dataPipeline = DataPipeline()
    dataPipeline.run_data_pipeline()