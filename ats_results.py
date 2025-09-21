import sqlite3
from datetime import datetime

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect("ats_results.db")
c = conn.cursor()

# Create table
c.execute('''
CREATE TABLE IF NOT EXISTS ats_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_filename TEXT,
    jd_filename TEXT,
    resume_text TEXT,
    jd_text TEXT,
    result_json TEXT,
    created_at TEXT
)
''')
conn.commit()

def save_ats_result(resume_file, jd_file, resume_text, jd_text, result_dict):
    c.execute('''
    INSERT INTO ats_results (resume_filename, jd_filename, resume_text, jd_text, result_json, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        resume_file.name,
        jd_file.name,
        resume_text,
        jd_text,
        json.dumps(result_dict),  # Store JSON as string
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
