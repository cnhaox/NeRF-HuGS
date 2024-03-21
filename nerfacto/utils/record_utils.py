from pathlib import Path
import logging
from torch.utils.tensorboard import SummaryWriter

class Recorder():
    def __init__(self, folder_path: Path) -> None:
        self.root = folder_path
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(self.root / 'run_log.log')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s # %(message)s')
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)
        self.writer = SummaryWriter(self.root)
    
    def print(self, message: str):
        print(message)
        self.logger.info(message)

    def __del__(self):
        self.writer.close()