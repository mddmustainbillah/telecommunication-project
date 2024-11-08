from pathlib import Path
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from telecommunication.config import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, logger
from utils import load_params


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the DataFrame by handling missing values, converting datatypes, and filtering data
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



def split_and_save_data(df: pd.DataFrame) -> None:
    """
    Split the dataframe into training and testing sets and save them to separate files
    """
    # Load parameters
    params = load_params()
    
    # Access split parameters
    test_size = params['split']['test_size']
    random_state = params['split']['random_state']
    
    df = df.copy()
    
    # Use parameters in the split
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state
    )

    return train_df, test_df




def main():
    """
    Main function to execute data cleaning, feature engineering, and mapping
    """
    try:  
        logger.info("Starting features generation and mapping...")
        logger.info(f"Reading raw data from {RAW_DATA_DIR / 'raw_dataset.csv'}")
        df = pd.read_csv(RAW_DATA_DIR / "raw_dataset.csv")

        logger.info("Cleaning data...")
        df = clean_data(df)

        logger.info("Engineering features...")
        df = feature_engineering(df)

        logger.info("Mapping features...")
        df = feature_mapping(df)

        logger.info(f"Saving interim data to {INTERIM_DATA_DIR / 'interim_dataset.csv'}")
        df.to_csv(INTERIM_DATA_DIR / "interim_dataset.csv", index=False)

        logger.info("Features generation and mapping completed successfully.")

        logger.info("Splitting data into training and testing sets...")
        train_df, test_df = split_and_save_data(df)   

        logger.info(f"Saving training and testing sets to {PROCESSED_DATA_DIR}")
        train_df.to_csv(PROCESSED_DATA_DIR / "train_dataset.csv", index=False)
        test_df.to_csv(PROCESSED_DATA_DIR / "test_dataset.csv", index=False)

        logger.info("Data splitting and saving completed successfully.")   


    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()