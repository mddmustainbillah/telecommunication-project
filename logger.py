import logging
import sys
from pathlib import Path
from typing import Optional
try:
    from tqdm import tqdm
    TQDM_INSTALLED = True
except ImportError:
    TQDM_INSTALLED = False

class TqdmStreamHandler(logging.StreamHandler):
    """Custom StreamHandler for tqdm compatibility"""
    def emit(self, record):
        try:
            msg = self.format(record)
            if TQDM_INSTALLED:
                tqdm.write(msg)
            else:
                self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logger(proj_root: Optional[Path] = None) -> logging.Logger:
    """
    Configure logger to write to both file and console
    
    Args:
        proj_root (Optional[Path]): Project root directory. If None, will determine automatically.
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Determine project root if not provided
    if proj_root is None:
        proj_root = Path(__file__).resolve().parents[1]  # Go up two levels from logger.py
    
    # Create logs directory in project root
    logs_dir = proj_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = logs_dir / "pipeline.log"
    
    # Create logger
    logger = logging.getLogger('telco_pipeline')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Create console handler with tqdm compatibility
    console_handler = TqdmStreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger