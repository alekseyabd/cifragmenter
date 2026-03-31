import logging
import warnings
from rdkit import RDLogger
import os
from logging.handlers import RotatingFileHandler


def setup_logging(level: str):
    warnings.filterwarnings("ignore")

    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level))

    logging.getLogger("pymatgen").setLevel(logging.ERROR)
    for name in list(logging.Logger.manager.loggerDict):
        if name.startswith("pymatgen"):
            logging.getLogger(name).setLevel(logging.ERROR)

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")

    try:
        from openbabel import openbabel as ob
        ob.obErrorLog.SetOutputLevel(0)
    except Exception:
        pass

import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Отключаем пропагацию в корневой логгер — это предотвращает вывод в консоль
    logger.propagate = False
    
    # Очищаем старые обработчики (на случай повторного вызова)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Создание директории для логов
    os.makedirs('logs', exist_ok=True)

    # Обработчик с ротацией (5 МБ, 3 файла)
    handler = RotatingFileHandler(
        f'logs/{log_file}',
        maxBytes=5*1024*1024,
        backupCount=3,
        encoding='utf-8'  # Корректная кодировка для кириллицы
    )

    # Формат сообщений
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger