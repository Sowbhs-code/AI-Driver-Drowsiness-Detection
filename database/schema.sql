-- ============================================================
-- AI-Based Driver Drowsiness & Distraction Detection System
-- SQL Schema (SQLite compatible)
-- Run: sqlite3 database/drowsiness.db < database/schema.sql
-- ============================================================

PRAGMA foreign_keys = ON;

-- ─── Session Records ──────────────────────────────────────────────────────────
-- One row per monitoring session (camera ON → camera OFF)
CREATE TABLE IF NOT EXISTS session_records (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time          DATETIME NOT NULL DEFAULT (datetime('now')),
    end_time            DATETIME,
    duration_secs       REAL    DEFAULT 0.0,

    -- Aggregate counters (updated when session ends)
    total_frames        INTEGER DEFAULT 0,
    drowsy_frames       INTEGER DEFAULT 0,
    distracted_frames   INTEGER DEFAULT 0,
    yawn_count          INTEGER DEFAULT 0,
    phone_detected_cnt  INTEGER DEFAULT 0,
    alert_count         INTEGER DEFAULT 0
);

-- ─── Detection Logs ───────────────────────────────────────────────────────────
-- One row per individual detection event during a session
CREATE TABLE IF NOT EXISTS detection_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES session_records(id) ON DELETE CASCADE,
    timestamp       DATETIME NOT NULL DEFAULT (datetime('now')),

    -- Detection type: DROWSY | YAWN | DISTRACTED | PHONE | HEAD_TURN
    detection_type  TEXT NOT NULL CHECK(detection_type IN
                    ('DROWSY','YAWN','DISTRACTED','PHONE','HEAD_TURN','NO_FACE')),

    -- Raw metric values captured at time of detection
    ear             REAL DEFAULT 0.0,   -- Eye Aspect Ratio
    mar             REAL DEFAULT 0.0,   -- Mouth Aspect Ratio
    head_pitch      REAL DEFAULT 0.0,   -- degrees
    head_yaw        REAL DEFAULT 0.0,   -- degrees
    head_roll       REAL DEFAULT 0.0,   -- degrees
    confidence      REAL DEFAULT 1.0,   -- YOLO detection confidence

    -- Severity: LOW | MEDIUM | HIGH | CRITICAL
    severity        TEXT DEFAULT 'HIGH' CHECK(severity IN ('LOW','MEDIUM','HIGH','CRITICAL')),

    -- Path to screenshot captured at event time (nullable)
    screenshot_path TEXT
);

-- Index for fast time-range queries
CREATE INDEX IF NOT EXISTS idx_detection_logs_timestamp
    ON detection_logs(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_detection_logs_session
    ON detection_logs(session_id);

CREATE INDEX IF NOT EXISTS idx_detection_logs_type
    ON detection_logs(detection_type);

-- ─── Alert Records ────────────────────────────────────────────────────────────
-- One row per audible/visual alert that was triggered
CREATE TABLE IF NOT EXISTS alert_records (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER NOT NULL REFERENCES session_records(id) ON DELETE CASCADE,
    timestamp    DATETIME NOT NULL DEFAULT (datetime('now')),

    -- Alert type: DROWSY | YAWN | DISTRACTED | PHONE
    alert_type   TEXT NOT NULL,

    audio_played INTEGER DEFAULT 0,   -- 0=false, 1=true
    tts_played   INTEGER DEFAULT 0,
    acknowledged INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_alert_records_timestamp
    ON alert_records(timestamp DESC);

-- ─── Driver Profiles ──────────────────────────────────────────────────────────
-- Optional per-driver calibration thresholds
CREATE TABLE IF NOT EXISTS driver_profiles (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL DEFAULT 'Default Driver',
    created_at          DATETIME DEFAULT (datetime('now')),

    -- Custom thresholds (NULL = use system defaults)
    ear_threshold       REAL,
    mar_threshold       REAL,
    head_pitch_thresh   REAL,
    head_yaw_thresh     REAL,

    -- Baseline EAR when driver is fully alert (calibrated)
    baseline_ear        REAL,

    is_active           INTEGER DEFAULT 1
);

-- Insert a default driver profile
INSERT OR IGNORE INTO driver_profiles (id, name, is_active)
    VALUES (1, 'Default Driver', 1);

-- ─── System Config ────────────────────────────────────────────────────────────
-- Key-value table for runtime configuration
CREATE TABLE IF NOT EXISTS system_config (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key         TEXT UNIQUE NOT NULL,
    value       TEXT NOT NULL,
    description TEXT,
    updated_at  DATETIME DEFAULT (datetime('now'))
);

-- Seed default configuration values
INSERT OR IGNORE INTO system_config (key, value, description) VALUES
    ('EAR_THRESHOLD',     '0.25', 'Eye Aspect Ratio threshold – below this = eyes closed'),
    ('MAR_THRESHOLD',     '0.6',  'Mouth Aspect Ratio threshold – above this = yawning'),
    ('EAR_CONSEC_FRAMES', '20',   'Consecutive closed-eye frames before drowsiness alert'),
    ('MAR_CONSEC_FRAMES', '15',   'Consecutive yawn frames before yawn alert'),
    ('HEAD_PITCH_THRESH', '20',   'Head pitch threshold in degrees'),
    ('HEAD_YAW_THRESH',   '30',   'Head yaw threshold in degrees'),
    ('PHONE_CONF_THRESH', '0.5',  'YOLOv8 minimum confidence for phone detection'),
    ('ALERT_COOLDOWN',    '5',    'Minimum seconds between repeated alerts of same type'),
    ('SCREENSHOT_SAVE',   '1',    'Whether to save screenshots on alert (1=yes, 0=no)'),
    ('TTS_ENABLED',       '1',    'Text-to-speech alerts enabled (1=yes, 0=no)');

-- ─── Views for Reporting ──────────────────────────────────────────────────────

-- Daily detection summary view
CREATE VIEW IF NOT EXISTS daily_summary AS
SELECT
    date(timestamp)           AS day,
    detection_type,
    COUNT(*)                  AS event_count,
    AVG(ear)                  AS avg_ear,
    AVG(mar)                  AS avg_mar,
    SUM(CASE WHEN severity = 'CRITICAL' THEN 1 ELSE 0 END) AS critical_count
FROM detection_logs
GROUP BY date(timestamp), detection_type
ORDER BY day DESC;

-- Session performance view
CREATE VIEW IF NOT EXISTS session_performance AS
SELECT
    s.id,
    s.start_time,
    s.duration_secs,
    s.alert_count,
    COUNT(d.id)               AS total_events,
    SUM(CASE WHEN d.detection_type = 'DROWSY'     THEN 1 ELSE 0 END) AS drowsy_events,
    SUM(CASE WHEN d.detection_type = 'YAWN'       THEN 1 ELSE 0 END) AS yawn_events,
    SUM(CASE WHEN d.detection_type = 'DISTRACTED' THEN 1 ELSE 0 END) AS distracted_events,
    SUM(CASE WHEN d.detection_type = 'PHONE'      THEN 1 ELSE 0 END) AS phone_events
FROM session_records s
LEFT JOIN detection_logs d ON d.session_id = s.id
GROUP BY s.id
ORDER BY s.start_time DESC;
