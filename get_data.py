import sqlite3
import pandas as pd

# Database path
db_path = "data/database.db"

# Output file path
output_file_path = "data/raw/data.csv"

# Connect to sqlite database
conn = sqlite3.connect(db_path)

# Query to fetch the data.
query = "SELECT * FROM telco"

# Read the query result into DataFrame
data = pd.read_sql(query, conn)

# Close the connection
conn.close()

# Save the data as a csv file
data.to_csv(output_file_path, index=False)

print(f"Data has been successfully saved to {output_file_path}")