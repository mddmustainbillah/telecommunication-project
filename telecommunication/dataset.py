import sqlite3
import pandas as pd
from typing import Optional

from telecommunication.config import logger


def fetch_telco_data(db_path: str = "../data/database.db") -> pd.DataFrame:
    """
    Fetch telco data from SQLite database.
    
    Args:
        db_path (str): Path to the SQLite database file
        
    Returns:
        pd.DataFrame: DataFrame containing the telco data
        
    Raises:
        sqlite3.Error: If there's an error connecting to or querying the database
    """
    try:
        # Connect to sqlite database
        conn = sqlite3.connect(db_path)
        logger.info(f"Connected to database at {db_path}")
        
        # Query to fetch the data
        query = """SELECT "Bundle Type",operator, "Product Name", validity, regularPrice, sellingPrice, commission FROM telco"""
        
        # Read the query result into DataFrame
        data = pd.read_sql(query, conn)
        logger.info(f"Data fetched successfully. Shape: {data.shape}")
        
        return data
    
    except sqlite3.Error as e:
        logger.error(f"Database error: {str(e)}")
        raise
    
    finally:
        if conn:
            conn.close()

def save_data(data: pd.DataFrame, output_path: str = "../data/raw/raw_dataset.csv") -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        data (pd.DataFrame): DataFrame to save
        output_path (str): Path where to save the CSV file
    """
    data.to_csv(output_path, index=False)
    logger.info(f"Data has been successfully saved to {output_path}")

def main():
    """Main function to execute data fetching and saving"""
    try:
        # Step 1: Fetch data
        logger.info("Fetching data from database...")
        df = fetch_telco_data()
        
        # Step 2: Save raw data
        logger.info("Saving raw data...")
        save_data(df)
        
        logger.info("Data pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()