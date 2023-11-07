from loguru import logger
from datetime import datetime
import os
import sys
from path_config import *


def setup_logger(log_file_name):
    #  Set up logging
    log_full_path = os.path.join(LOG_DIR, log_file_name)
    print(log_full_path)

    #  Clear previous log
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <red>{level}</red> {message}",
               colorize=True, )
    logger.add(log_full_path, format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}")


def logd(message):
    logger.debug(message)


def loge(message):
    logger.error(message)


def logi(message):
    logger.info(message)


def logw(message):
    logger.warning(message)