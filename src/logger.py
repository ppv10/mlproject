import logging
from datetime import datetime
from pathlib import Path

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = Path(__file__).resolve().parents[1] / 'logs'
logs_path.mkdir(exist_ok=True)

LOG_FILE_PATH = logs_path / LOG_FILE

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,


)
