import sqlite3
import pandas as pd

DB_PATH = "school_system.db"
conn = sqlite3.connect(DB_PATH)

# Show all tables
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
print("📌 Tables in DB:", tables)

# View Users
users_df = pd.read_sql("SELECT * FROM users", conn)
print("\n👤 Users Table:\n", users_df)

# View Predictions
preds_df = pd.read_sql("SELECT * FROM predictions", conn)
print("\n📊 Predictions Table:\n", preds_df)

conn.close()
