import logging
import warnings
from rdkit import RDLogger

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
