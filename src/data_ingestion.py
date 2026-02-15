import os
import subprocess
import shutil

from utils.custom_exception import CustomException
from utils.logger import get_logger
from config.paths_config import *

logger = get_logger(__name__)


class DataIngestion:
    def __init__(
        self,
        repo_url: str,
        repo_dir: str,
        output_dir: str
    ):
        self.repo_url = repo_url
        self.repo_dir = repo_dir
        self.output_dir = output_dir

    logger.info("Data Ingestion Initiated.")

    def clone_or_pull_repo(self):
        """Clone the repository if not present, otherwise pull latest changes."""
        try:

            if os.path.exists(self.repo_dir):
                logger.info(f"[INFO] Repo already exists. Pulling latest changes in {self.repo_dir}...")
                subprocess.run(["git", "-C", self.repo_dir, "pull"], check=True)
            else:
                logger.info(f"[INFO] Cloning repo: {self.repo_url}")
                subprocess.run(["git", "clone", self.repo_url, self.repo_dir], check=True)
        
        except Exception as e:
            logger.error(f"Error while cloning repo - {e}")
            raise CustomException(f"Failed to clone repo : ", e)
        

    def create_output_folder(self):
        """Create raw-word-list directory fresh."""
        try:

            if os.path.exists(self.output_dir):
                logger.info(f"[INFO] Removing existing output folder: {self.output_dir}")
                shutil.rmtree(self.output_dir)

            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"[INFO] Created output folder: {self.output_dir}")
        
        except Exception as e:
            logger.error(f"Error while creating output folder - {e}")
            raise CustomException(f"Failed to create output folder : ", e)
        


    def copy_repo_folders(self):
        """
        Copy all folders and files from cloned repo into raw-word-list directory.
        Excludes .git folder.
        """
        try:
            logger.info(f"[INFO] Copying repo contents into: {self.output_dir}")

            for item in os.listdir(self.repo_dir):
                source_path = os.path.join(self.repo_dir, item)

                # skip git metadata
                if item == ".git":
                    continue

                dest_path = os.path.join(self.output_dir, item)

                if os.path.isdir(source_path):
                    shutil.copytree(source_path, dest_path)
                else:
                    shutil.copy2(source_path, dest_path)

            logger.info("[SUCCESS] All folders copied into raw-word-list successfully!")
        
        except Exception as e:
            logger.error(f"Error while copying repo folders - {e}")
            raise CustomException(f"Failed to copy repo folders : ", e)
        

    def run(self):
        self.clone_or_pull_repo()
        self.create_output_folder()
        self.copy_repo_folders()


