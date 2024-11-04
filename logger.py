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

def setup_logger(log_file: str = "logs/pipeline.log") -> logging.Logger:
    """
    Configure logger to write to both file and console
    
    Args:
        log_file (str): Path to the log file
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('telco_pipeline')
    logger.setLevel(logging.INFO)
    
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