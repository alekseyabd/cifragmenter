from __future__ import annotations

from pathlib import Path
import typer

import os

from .logging_conf import setup_logging
from .logging_conf import setup_logger
from .runner import run as run_job

app = typer.Typer(add_completion=False)

@app.command()
def run(
    input_file: Path = typer.Argument(..., help="Путь до входного файла"),
    log_level: str = typer.Option("INFO", "--log-level", help="DEBUG/INFO/WARNING/ERROR"),
    ccdc_chemical_name_systematic: str = typer.Option(
        "_chemical_name_systematic",
        "--ccdc-chemical-name",
        help="Line with chemical name"
    ),
    db_code_pattern: str = typer.Option(
        "_database_code_depnum_ccdc_archive",
        "--db-code-pattern",
        help="Line with RefCode"
    ),
    min_occ: float = typer.Option(
        0.5,
        "--min-occ",
        help="Minimal value of occupancy"
    ),
    fragment_type: str = typer.Option(
        "coord",
        "--fragment-type",
        help="Type of fragmentation: coord/mols"
    ),
    property: str = typer.Option(
        "meelting_point",
        "--property",
        help="Property for search in cif"
    ),
    timeout: int = typer.Option(
        3000,
        "--timeout",
        help="Maximum file processing time"
    ),
    n_jobs: int = typer.Option(
        os.cpu_count()-1,
        "--n_jobs",
        help="Number of processor cores"
    )
):
    #setup_logging(log_level)
    #logger = setup_logger('my_service', 'service.log')
    #logger.info('Сервис запущен')
    code = run_job(
    	input_file=input_file,
        ccdc_chemical_name_systematic=ccdc_chemical_name_systematic,
        db_code_pattern=db_code_pattern,
        min_occ=min_occ,
        fragment_type=fragment_type,
        property=property,
        TIMEOUT=timeout,
        n_jobs=n_jobs,
        log_level=log_level)
    raise typer.Exit(code=code)

if __name__ == "__main__":
    app()

