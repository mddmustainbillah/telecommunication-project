from pathlib import Path
from dotenv import load_dotenv
from logger import setup_logger 

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

# Initialize logger with project root
logger = setup_logger(proj_root=PROJ_ROOT)

# Load environment variables from .env file if it exists
load_dotenv()

# Log project root path
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Rest of your directory configurations...
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MLRUNS_DIR = PROJ_ROOT / "mlruns"
LOGS_DIR = PROJ_ROOT / "logs"

# Ensure directories exist
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
