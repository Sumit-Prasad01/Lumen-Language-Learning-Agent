import os
import subprocess
import shutil


class DataIngestion:
    def __init__(
        self,
        repo_url: str,
        repo_dir: str = "all-words-in-all-languages",
        output_dir: str = "raw-word-list"
    ):
        self.repo_url = repo_url
        self.repo_dir = repo_dir
        self.output_dir = output_dir

    def clone_or_pull_repo(self):
        """Clone the repository if not present, otherwise pull latest changes."""
        if os.path.exists(self.repo_dir):
            print(f"[INFO] Repo already exists. Pulling latest changes in {self.repo_dir}...")
            subprocess.run(["git", "-C", self.repo_dir, "pull"], check=True)
        else:
            print(f"[INFO] Cloning repo: {self.repo_url}")
            subprocess.run(["git", "clone", self.repo_url, self.repo_dir], check=True)

    def create_output_folder(self):
        """Create raw-word-list directory fresh."""
        if os.path.exists(self.output_dir):
            print(f"[INFO] Removing existing output folder: {self.output_dir}")
            shutil.rmtree(self.output_dir)

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[INFO] Created output folder: {self.output_dir}")

    def copy_repo_folders(self):
        """
        Copy all folders and files from cloned repo into raw-word-list directory.
        Excludes .git folder.
        """
        print(f"[INFO] Copying repo contents into: {self.output_dir}")

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

        print("[SUCCESS] All folders copied into raw-word-list successfully!")

    def run(self):
        self.clone_or_pull_repo()
        self.create_output_folder()
        self.copy_repo_folders()


if __name__ == "__main__":
    REPO_URL = "https://github.com/eymenefealtun/all-words-in-all-languages.git"

    ingestion = DataIngestion(repo_url=REPO_URL)
    ingestion.run()
