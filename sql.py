import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime

# ==== CONFIGURATION ====
EXCEL_PATH = "FINAL-AI-27-9-2025.xlsx"  # path to your Excel sheet
SHEET_NAME = 0  # or sheet name as string
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "dbname": "maternal_db",
    "user": "postgres",
    "password": "postgres",
}
TABLE_NAME = "patients_patient"

# ==== STEP 1: READ EXCEL ====
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
df = df.rename(columns=str.strip)  # clean up column names
expected_cols = {"date_of_admission", "baby_file_number", "mother_file_number"}
if not expected_cols.issubset(df.columns):
    raise ValueError(f"Excel must contain columns: {expected_cols}")

# Normalize date and add 00:00:00 time
df["date_of_admission"] = pd.to_datetime(df["date_of_admission"]).dt.normalize()

# ==== STEP 2: CONNECT TO POSTGRES ====
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()
def clean_number(val):
    if pd.isna(val):
        return None
    # Convert numeric like 470843.0 → '470843'
    if isinstance(val, (int, float)):
        return str(int(val))
    # Trim spaces
    return str(val).strip()
# ==== STEP 3: PREPARE UPDATES ====
updates = []
for _, row in df.iterrows():
    date_time = row["date_of_admission"]
    baby_file = clean_number(row["baby_file_number"])
    mother_file = clean_number(row["mother_file_number"])
    print(mother_file)
    if pd.notna(mother_file) and pd.notna(baby_file):
        updates.append((baby_file, date_time, mother_file))

# ==== STEP 4: EXECUTE UPDATES ====
query = f"""
    UPDATE {TABLE_NAME}
    SET baby_file_number = %s,
        time_of_admission = %s
    WHERE file_number = %s
"""
execute_batch(cur, query, updates, page_size=100)
conn.commit()

# ==== STEP 5: CLEANUP ====
print(f"✅ Updated {cur.rowcount} rows successfully.")
cur.close()
conn.close()