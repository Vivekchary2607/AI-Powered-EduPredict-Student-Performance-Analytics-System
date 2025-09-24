# init_db.py
from database import init_db, add_user

if __name__ == "__main__":
    # Initialize database
    init_db()

    # Create sample users (you can change usernames and passwords later)
    add_user("admin", "admin123", role="admin", subject="None")   # Admin has no subject
    add_user("teacher1", "teach123", role="teacher", subject="Math")  # Teacher with subject Math

    print("✅ Database initialized and sample users added:")
    print("   - Admin -> username: admin | password: admin123 | role: admin")
    print("   - Teacher -> username: teacher1 | password: teach123 | role: teacher (Math)")
