from pathlib import Path
from tqdm import tqdm
import pandas as pd

from telecommunication.config import PROCESSED_DATA_DIR

from logger import setup_logger

# Initialize logger
logger = setup_logger()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataframe by handling missing values, converting datatypes, and filtering data
    """
    df = df.copy()
    
    # Convert price columns to numeric, removing commas
    for col in ['sellingPrice', 'regularPrice']:
        df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
    
    df['commission'] = pd.to_numeric(df['commission'], errors='coerce')
    
    # Drop any rows with NaN values created during numeric conversion
    df = df.dropna(subset=['sellingPrice', 'regularPrice', 'commission']).reset_index(drop=True)
    
    df['Bundle Type'] = df['Bundle Type'].str.replace('Intenet & Minute', 'Internet & Minute')
    
    # Drop rows where either regularPrice OR sellingPrice is 0
    mask = (df['regularPrice'] == 0) | (df['sellingPrice'] == 0)
    df = df[~mask].reset_index(drop=True)
    
    # Ensure positive values
    df['sellingPrice'] = df['sellingPrice'].abs()
    df['commission'] = df['commission'].abs()
    
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from Product Name and process validity
    
    Args:
        df: Input DataFrame
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Extract Internet data
    def extract_internet(x):
        try:
            value = x.split()[0]
            unit = x.split()[1]
            if 'GB' in unit:
                return float(value.split('/')[0])  # Handle cases like '40/45'
            elif 'MB' in unit:
                return float(value.split('/')[0]) / 1024  # Convert MB to GB
            return 0
        except (IndexError, ValueError):
            return 0
    
    # Process Internet column
    df['Internet'] = (df['Product Name']
                     .str.split()
                     .str[0:2]
                     .str.join(' ')
                     .apply(extract_internet))
    
    # Extract Minutes
    def extract_minutes(x):
        try:
            if 'min' in x:
                return float(x.split()[0])
            return 0
        except (IndexError, ValueError):
            return 0
    
    # Process Minutes column
    df['Minutes'] = (df['Product Name']
                    .str.split()
                    .str[-2:]
                    .str.join(' ')
                    .apply(extract_minutes))
    
    # Convert validity to float
    df['validity'] = pd.to_numeric(df['validity'].str.split().str[0], errors='coerce')
    
    # Drop rows with NaN values in the new columns
    df = df.dropna(subset=['Internet', 'Minutes', 'validity']).reset_index(drop=True)
    
    # Drop Product Name column
    df.drop(columns=['Product Name'], inplace=True)
    
    return df


def feature_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map features to their respective categories
    """
    df = df.copy()

    bundle_type_mapping = {'Internet & Minute': 1, 'Internet': 2, 'Minute': 3}
    df['Bundle Type'] = df['Bundle Type'].map(bundle_type_mapping)
    
    operator_mapping = {'robi': 1, 'gp': 2, 'airtel': 3, 'bl': 4}
    df['operator'] = df['operator'].map(operator_mapping)
    
    return df




@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
