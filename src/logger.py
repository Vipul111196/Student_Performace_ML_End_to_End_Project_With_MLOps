import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok= True)

LOG_FILE = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d - %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    logging.info('Logging has just started')
