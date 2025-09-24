import sqlite3

DB_PATH = "school_system.db"
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("ALTER TABLE users ADD COLUMN subject TEXT DEFAULT 'General'")
conn.commit()
conn.close()
print("✅ Subject column added to users table.")
