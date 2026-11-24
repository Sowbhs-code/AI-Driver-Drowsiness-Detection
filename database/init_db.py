"""
init_db.py
─────────────────────────────────────────────────────────
Standalone script to initialise (or re-initialise) the
SQLite database from the raw SQL schema file.

Usage
─────
    python database/init_db.py              # creates drowsiness.db
    python database/init_db.py --reset      # drops and recreates all tables
─────────────────────────────────────────────────────────
"""

import os
import sys
import sqlite3
import argparse

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH     = os.path.join(BASE_DIR, 'database', 'drowsiness.db')
SCHEMA_PATH = os.path.join(BASE_DIR, 'database', 'schema.sql')


def init_database(reset: bool = False) -> None:
    """Create (or reset) the SQLite database."""

    if reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"[RESET] Removed existing database: {DB_PATH}")

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
        sql = f.read()

    # Execute all statements
    cursor.executescript(sql)
    conn.commit()
    conn.close()

    print(f"[OK] Database initialised at: {DB_PATH}")
    _verify_tables()


def _verify_tables() -> None:
    """Print a summary of created tables and row counts."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cursor.fetchall()]

    print("\n── Tables Created ──────────────────────────────────────")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]
        print(f"  {table:<30} {count:>4} rows")

    cursor.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name;")
    views = [row[0] for row in cursor.fetchall()]
    if views:
        print("\n── Views Created ───────────────────────────────────────")
        for view in views:
            print(f"  {view}")

    print("────────────────────────────────────────────────────────\n")
    conn.close()


def insert_sample_data() -> None:
    """Insert sample/demo data for UI testing."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    from datetime import datetime, timedelta
    import random

    # Sample sessions
    now = datetime.utcnow()
    for i in range(5):
        start = now - timedelta(hours=i * 2 + random.randint(0, 3))
        end   = start + timedelta(minutes=random.randint(15, 60))
        cursor.execute("""
            INSERT INTO session_records
                (start_time, end_time, duration_secs, total_frames,
                 drowsy_frames, distracted_frames, yawn_count, alert_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (start, end, (end - start).total_seconds(),
              random.randint(500, 3000),
              random.randint(10, 80),
              random.randint(5, 40),
              random.randint(2, 10),
              random.randint(3, 15)))

    conn.commit()
    session_ids = [row[0] for row in cursor.execute("SELECT id FROM session_records").fetchall()]

    # Sample detection logs
    detection_types = ['DROWSY', 'YAWN', 'DISTRACTED', 'PHONE', 'HEAD_TURN']
    severities      = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

    for _ in range(50):
        ts      = now - timedelta(minutes=random.randint(0, 300))
        sid     = random.choice(session_ids)
        dtype   = random.choice(detection_types)
        sev     = random.choices(severities, weights=[10, 30, 40, 20])[0]
        ear     = round(random.uniform(0.10, 0.35), 3)
        mar     = round(random.uniform(0.20, 0.90), 3)
        cursor.execute("""
            INSERT INTO detection_logs
                (session_id, timestamp, detection_type, ear, mar,
                 head_pitch, head_yaw, confidence, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (sid, ts, dtype, ear, mar,
              round(random.uniform(-30, 30), 1),
              round(random.uniform(-40, 40), 1),
              round(random.uniform(0.5, 1.0), 2),
              sev))

    # Sample alerts
    for _ in range(20):
        ts  = now - timedelta(minutes=random.randint(0, 300))
        sid = random.choice(session_ids)
        cursor.execute("""
            INSERT INTO alert_records
                (session_id, timestamp, alert_type, audio_played, tts_played)
            VALUES (?, ?, ?, ?, ?)
        """, (sid, ts, random.choice(['DROWSY', 'YAWN', 'DISTRACTED']),
              random.randint(0, 1), random.randint(0, 1)))

    conn.commit()
    conn.close()
    print("[OK] Sample data inserted.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Initialise the drowsiness detection database.')
    parser.add_argument('--reset',  action='store_true', help='Drop and recreate all tables')
    parser.add_argument('--sample', action='store_true', help='Insert sample demo data')
    args = parser.parse_args()

    init_database(reset=args.reset)

    if args.sample:
        insert_sample_data()
