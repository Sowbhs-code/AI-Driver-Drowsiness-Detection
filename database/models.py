"""
models.py
─────────────────────────────────────────────────────────
SQLAlchemy ORM models for the drowsiness detection system.

Tables
──────
    session_records   – one row per monitoring session
    detection_logs    – every individual detection event
    alert_records     – every alert that was triggered
    driver_profiles   – optional per-driver calibration
    system_config     – key-value store for runtime config

Schema is auto-created by Flask-SQLAlchemy via db.create_all()
─────────────────────────────────────────────────────────
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


# ─── User Account ─────────────────────────────────────────────────────────────

class User(UserMixin, db.Model):
    """Login account for the system."""
    __tablename__ = 'users'

    id           = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username     = db.Column(db.String(64), unique=True, nullable=False)
    email        = db.Column(db.String(128), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    full_name    = db.Column(db.String(128), nullable=True)
    role         = db.Column(db.String(16), default='driver')   # driver | admin
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    last_login   = db.Column(db.DateTime, nullable=True)
    is_active    = db.Column(db.Boolean, default=True)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def to_dict(self) -> dict:
        return {
            'id':        self.id,
            'username':  self.username,
            'email':     self.email,
            'full_name': self.full_name,
            'role':      self.role,
        }

    def __repr__(self):
        return f'<User {self.username}>'


# ─── Session Record ───────────────────────────────────────────────────────────

class SessionRecord(db.Model):
    """
    One row per monitoring session (camera start → camera stop).
    """
    __tablename__ = 'session_records'

    id             = db.Column(db.Integer, primary_key=True, autoincrement=True)
    start_time     = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    end_time       = db.Column(db.DateTime, nullable=True)
    duration_secs  = db.Column(db.Float, default=0.0)

    # Aggregate counters
    total_frames       = db.Column(db.Integer, default=0)
    drowsy_frames      = db.Column(db.Integer, default=0)
    distracted_frames  = db.Column(db.Integer, default=0)
    yawn_count         = db.Column(db.Integer, default=0)
    phone_detected_cnt = db.Column(db.Integer, default=0)
    alert_count        = db.Column(db.Integer, default=0)

    # Relationships
    detection_logs = db.relationship('DetectionLog',  back_populates='session', cascade='all, delete-orphan')
    alert_records  = db.relationship('AlertRecord',   back_populates='session', cascade='all, delete-orphan')

    def to_dict(self) -> dict:
        return {
            'id':               self.id,
            'start_time':       self.start_time.isoformat() if self.start_time else None,
            'end_time':         self.end_time.isoformat()   if self.end_time   else None,
            'duration_secs':    self.duration_secs,
            'total_frames':     self.total_frames,
            'drowsy_frames':    self.drowsy_frames,
            'distracted_frames':self.distracted_frames,
            'yawn_count':       self.yawn_count,
            'alert_count':      self.alert_count,
        }

    def __repr__(self):
        return f'<Session id={self.id} start={self.start_time}>'


# ─── Detection Log ────────────────────────────────────────────────────────────

class DetectionLog(db.Model):
    """
    One row per detection event captured during a session.
    """
    __tablename__ = 'detection_logs'

    id              = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id      = db.Column(db.Integer, db.ForeignKey('session_records.id'), nullable=False)
    timestamp       = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Detection type: DROWSY | YAWN | DISTRACTED | PHONE | HEAD_TURN
    detection_type  = db.Column(db.String(32), nullable=False)

    # Raw metric values at time of detection
    ear             = db.Column(db.Float, default=0.0)   # Eye Aspect Ratio
    mar             = db.Column(db.Float, default=0.0)   # Mouth Aspect Ratio
    head_pitch      = db.Column(db.Float, default=0.0)   # degrees
    head_yaw        = db.Column(db.Float, default=0.0)   # degrees
    head_roll       = db.Column(db.Float, default=0.0)   # degrees
    confidence      = db.Column(db.Float, default=1.0)   # for YOLO detections

    # Severity: LOW | MEDIUM | HIGH | CRITICAL
    severity        = db.Column(db.String(16), default='HIGH')

    # Optional screenshot path
    screenshot_path = db.Column(db.String(256), nullable=True)

    # Relationship
    session = db.relationship('SessionRecord', back_populates='detection_logs')

    def to_dict(self) -> dict:
        return {
            'id':               self.id,
            'session_id':       self.session_id,
            'timestamp':        self.timestamp.isoformat() if self.timestamp else None,
            'detection_type':   self.detection_type,
            'ear':              self.ear,
            'mar':              self.mar,
            'head_pitch':       self.head_pitch,
            'head_yaw':         self.head_yaw,
            'head_roll':        self.head_roll,
            'confidence':       self.confidence,
            'severity':         self.severity,
            'screenshot_path':  self.screenshot_path,
        }

    def __repr__(self):
        return f'<DetectionLog id={self.id} type={self.detection_type} ts={self.timestamp}>'


# ─── Alert Record ─────────────────────────────────────────────────────────────

class AlertRecord(db.Model):
    """
    One row each time an audible / visual alert fires.
    """
    __tablename__ = 'alert_records'

    id           = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id   = db.Column(db.Integer, db.ForeignKey('session_records.id'), nullable=False)
    timestamp    = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Alert type: DROWSY | YAWN | DISTRACTED | PHONE
    alert_type   = db.Column(db.String(32), nullable=False)

    # How the alert was delivered
    audio_played = db.Column(db.Boolean, default=False)
    tts_played   = db.Column(db.Boolean, default=False)
    acknowledged = db.Column(db.Boolean, default=False)

    # Relationship
    session = db.relationship('SessionRecord', back_populates='alert_records')

    def to_dict(self) -> dict:
        return {
            'id':           self.id,
            'session_id':   self.session_id,
            'timestamp':    self.timestamp.isoformat() if self.timestamp else None,
            'alert_type':   self.alert_type,
            'audio_played': self.audio_played,
            'tts_played':   self.tts_played,
            'acknowledged': self.acknowledged,
        }

    def __repr__(self):
        return f'<AlertRecord id={self.id} type={self.alert_type}>'


# ─── Driver Profile ───────────────────────────────────────────────────────────

class DriverProfile(db.Model):
    """
    Optional per-driver calibration values (personalised thresholds).
    """
    __tablename__ = 'driver_profiles'

    id                  = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name                = db.Column(db.String(64), nullable=False, default='Default Driver')
    created_at          = db.Column(db.DateTime, default=datetime.utcnow)

    # Calibrated thresholds (NULL = use global defaults)
    ear_threshold       = db.Column(db.Float, nullable=True)
    mar_threshold       = db.Column(db.Float, nullable=True)
    head_pitch_thresh   = db.Column(db.Float, nullable=True)
    head_yaw_thresh     = db.Column(db.Float, nullable=True)

    # Baseline EAR when fully awake (calibrated per driver)
    baseline_ear        = db.Column(db.Float, nullable=True)

    is_active           = db.Column(db.Boolean, default=True)

    def to_dict(self) -> dict:
        return {
            'id':               self.id,
            'name':             self.name,
            'ear_threshold':    self.ear_threshold,
            'mar_threshold':    self.mar_threshold,
            'baseline_ear':     self.baseline_ear,
            'is_active':        self.is_active,
        }

    def __repr__(self):
        return f'<DriverProfile id={self.id} name={self.name}>'


# ─── System Config ────────────────────────────────────────────────────────────

class SystemConfig(db.Model):
    """
    Key-value store for runtime configuration overrides.
    """
    __tablename__ = 'system_config'

    id          = db.Column(db.Integer, primary_key=True, autoincrement=True)
    key         = db.Column(db.String(64), unique=True, nullable=False)
    value       = db.Column(db.String(256), nullable=False)
    description = db.Column(db.String(256), nullable=True)
    updated_at  = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<SystemConfig {self.key}={self.value}>'
