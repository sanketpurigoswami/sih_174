"""
storage.py — SQLite persistence for VIGIL mission logs + experiment scripts.

Tables:
  experiments  – JSON blobs of experiment phase scripts
  timeline     – every completed/failed step row
  mission      – singleton row tracking current mission state
"""

import sqlite3
import json
import os
import threading
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "vigil.db")
SEED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "experiments.json")

_local = threading.local()


def _get_conn():
    """One connection per thread (SQLite requirement)."""
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _local.conn = sqlite3.connect(DB_PATH)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn


def init_db():
    """Create tables and seed experiments if empty."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id   TEXT PRIMARY KEY,
            data TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS timeline (
            rowid     INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id TEXT NOT NULL,
            phase     TEXT NOT NULL,
            step      INTEGER NOT NULL,
            step_label TEXT NOT NULL,
            success   INTEGER NOT NULL DEFAULT 1,
            accuracy  REAL NOT NULL DEFAULT 0.0,
            start_time TEXT NOT NULL,
            end_time   TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS mission (
            id         INTEGER PRIMARY KEY CHECK (id = 1),
            status     TEXT NOT NULL DEFAULT 'idle',
            phase_id   TEXT,
            mission_id TEXT,
            started_at TEXT
        );
    """)
    # Always re-seed experiments from JSON so updates are picked up
    if os.path.exists(SEED_PATH):
        with open(SEED_PATH, "r", encoding="utf-8") as f:
            phases = json.load(f)
        for phase in phases:
            conn.execute(
                "INSERT OR REPLACE INTO experiments (id, data) VALUES (?, ?)",
                (phase["id"], json.dumps(phase)),
            )
    # Ensure mission singleton exists and is always reset to idle on server init
    conn.execute(
        """INSERT INTO mission (id, status, phase_id, mission_id, started_at) 
           VALUES (1, 'idle', NULL, NULL, NULL)
           ON CONFLICT(id) DO UPDATE SET status='idle', phase_id=NULL, mission_id=NULL, started_at=NULL"""
    )
    conn.commit()



# ── Experiments ──────────────────────────────────────────────────────────────

def get_experiments():
    conn = _get_conn()
    rows = conn.execute("SELECT data FROM experiments ORDER BY rowid").fetchall()
    return [json.loads(r["data"]) for r in rows]


def get_experiment(phase_id):
    conn = _get_conn()
    row = conn.execute("SELECT data FROM experiments WHERE id = ?", (phase_id,)).fetchone()
    return json.loads(row["data"]) if row else None


def add_experiment(phase_data):
    """Validate and insert a new experiment script. Returns (ok, error_msg)."""
    # Validate shape
    if not isinstance(phase_data, dict):
        return False, "Payload must be a JSON object."
    for key in ("id", "name", "steps"):
        if key not in phase_data:
            return False, f"Missing required key: '{key}'."
    if not isinstance(phase_data["steps"], list) or len(phase_data["steps"]) == 0:
        return False, "'steps' must be a non-empty array."
    for i, step in enumerate(phase_data["steps"]):
        for sk in ("id", "label", "expected_object", "expected_action"):
            if sk not in step:
                return False, f"Step {i} missing key: '{sk}'."
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO experiments (id, data) VALUES (?, ?)",
            (phase_data["id"], json.dumps(phase_data)),
        )
        conn.commit()
        return True, None
    except sqlite3.IntegrityError:
        return False, f"Experiment with id '{phase_data['id']}' already exists."


# ── Timeline ────────────────────────────────────────────────────────────────

def add_timeline_row(mission_id, phase, step, step_label, success, accuracy, start_time, end_time):
    conn = _get_conn()
    conn.execute(
        """INSERT INTO timeline
           (mission_id, phase, step, step_label, success, accuracy, start_time, end_time)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (mission_id, phase, step, step_label, int(success), accuracy, start_time, end_time),
    )
    conn.commit()


def get_timeline(mission_id=None):
    conn = _get_conn()
    if mission_id:
        rows = conn.execute(
            "SELECT * FROM timeline WHERE mission_id = ? ORDER BY rowid", (mission_id,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM timeline ORDER BY rowid").fetchall()
    return [dict(r) for r in rows]


def get_full_log(mission_id=None):
    """Return all timeline rows for download."""
    return get_timeline(mission_id)


def clear_timeline(mission_id):
    conn = _get_conn()
    conn.execute("DELETE FROM timeline WHERE mission_id = ?", (mission_id,))
    conn.commit()


def delete_experiment(phase_id):
    conn = _get_conn()
    conn.execute("DELETE FROM experiments WHERE id = ?", (phase_id,))
    conn.commit()
    return True


def get_timeline_csv(mission_id=None):
    """Return timeline rows formatted as CSV string."""
    rows = get_timeline(mission_id)
    headers = ["Mission ID", "Phase", "Step", "Step Label", "Success", "Accuracy (%)", "Start Time", "End Time"]
    lines = [",".join(headers)]
    for r in rows:
        acc_pct = f"{float(r.get('accuracy', 0)) * 100:.1f}"
        success_str = "SUCCESS" if r.get("success") else "FAILED"
        label = f'"{r.get("step_label", "").replace(chr(34), chr(39))}"'
        phase = f'"{r.get("phase", "").replace(chr(34), chr(39))}"'
        line = f'{r.get("mission_id", "")},{phase},{r.get("step", "")},{label},{success_str},{acc_pct},{r.get("start_time", "")},{r.get("end_time", "")}'
        lines.append(line)
    return "\n".join(lines)


# ── Mission state ────────────────────────────────────────────────────────────

def get_mission_state():
    conn = _get_conn()
    row = conn.execute("SELECT * FROM mission WHERE id = 1").fetchone()
    return dict(row) if row else {"status": "idle", "phase_id": None, "mission_id": None, "started_at": None}


def set_mission_state(status, phase_id=None, mission_id=None, started_at=None):
    conn = _get_conn()
    conn.execute(
        """UPDATE mission
           SET status = ?, phase_id = ?, mission_id = ?, started_at = ?
           WHERE id = 1""",
        (status, phase_id, mission_id, started_at),
    )
    conn.commit()

