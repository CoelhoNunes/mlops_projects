import logging
from pathlib import Path
import pandas as pd
from zenml import step

# Default: <repo_root>/src/data/olist_customers_dataset.csv
DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "olist_customers_dataset.csv"

# WSL fallback to your Windows path (adjust if your folder name differs)
WSL_FALLBACK = Path("/mnt/c/Users/Coelh/OneDrive/Desktop/CODING/mlops_projects/src/data/olist_customers_dataset.csv")

class IngestData:
    def __init__(self, csv_path: str | None = None) -> None:
        self.csv_path = Path(csv_path) if csv_path else DEFAULT_CSV

    def get_data(self) -> pd.DataFrame:
        path = self.csv_path
        if not path.exists():
            if WSL_FALLBACK.exists():
                path = WSL_FALLBACK
            else:
                raise FileNotFoundError(
                    f"CSV not found.\nTried:\n  - {self.csv_path}\n  - {WSL_FALLBACK}"
                )
        return pd.read_csv(path)

@step
def ingest_data(csv_path: str | None = None) -> pd.DataFrame:
    """
    Load the customers CSV. Optionally override the CSV path via `csv_path`.
    """
    try:
        ingestor = IngestData(csv_path)
        df = ingestor.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise