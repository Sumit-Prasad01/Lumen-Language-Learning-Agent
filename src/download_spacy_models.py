import sys
import subprocess
from utils.logger import get_logger
from config.models_list import SPACY_MODELS

logger = get_logger(__name__)


class DownloadSpacyModels:

    def __init__(self):
        self.spacy_models : dict = SPACY_MODELS

    
    def ensure_dependencies(self):
        """Ensure pip and spaCy are installed, then download required spaCy models."""
        def run_cmd(cmd):
            return subprocess.run(cmd, capture_output=True, text=True)
        

        pip_check = run_cmd([sys.executable, "-m", "pip", "--version"])
        if pip_check.returncode != 0:
            logger.info("pip not found, installing pip using ensurepip...")
            ensure = run_cmd([sys.executable, "-m", "ensurepip", "--upgrade"])
            logger.info(ensure.stdout)
            logger.info(ensure.stderr)
            upgrade = run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            logger.info(upgrade.stdout)
            logger.info(upgrade.stderr)
        
        
        spacy_check = run_cmd([sys.executable, "-c", "import spacy"])
        if spacy_check.returncode != 0:
            logger.info("spaCy not found, installing spaCy...")
            install_spacy = run_cmd([sys.executable, "-m", "pip", "install", "spacy"])
            logger.info(install_spacy.stdout)
            logger.info(install_spacy.stderr)
        
        
        for model in self.spacy_models.values():
            logger.info(f"Downloading spaCy model: {model}")
            result = run_cmd([sys.executable, "-m", "spacy", "download", model])
            logger.info(result.stdout)
            logger.info(result.stderr)

if __name__ == "__main__":

    downloader = DownloadSpacyModels()
    downloader.ensure_dependencies()