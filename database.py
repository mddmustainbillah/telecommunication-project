import sqlite3
import pandas as pd

# Database and csv file path
db_path = "data/database.db"
csv_path = "data/telco.csv"

# Connect to the SQLite database (it will be created if it doesn't exist)
conn = sqlite3.connect(db_path)

# Load the csv file into pandas DataFrame
data = pd.read_csv(csv_path)

# Push the DataFrame into the SQLite database, creating a table automatically
# 'telco' is the table name in SQLite, adjust as necessary
data.to_sql('telco', conn, if_exists='replace', index=False)

# Commit changes and close the connection
conn.commit()
conn.close()