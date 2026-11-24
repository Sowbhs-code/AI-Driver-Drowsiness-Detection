"""
db_manager.py
─────────────────────────────────────────────────────────
High-level database manager — wraps raw SQLAlchemy calls
so app.py and other modules never touch ORM internals.

All public methods accept / return plain Python dicts or
ORM model instances for maximum flexibility.
─────────────────────────────────────────────────────────
"""

from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy import func, desc

from database.models import db, SessionRecord, DetectionLog, AlertRecord, DriverProfile, SystemConfig


class DatabaseManager:
    """
    Centralised data-access layer for the drowsiness detection system.

    Parameters
    ----------
    app : Flask application instance (used for app_context pushes)
    """

    def __init__(self, app=None):
        self.app = app

    # ─── Session Management ───────────────────────────────────────────────────

    def create_session(self) -> SessionRecord:
        """Open a new monitoring session row and return it."""
        session = SessionRecord(start_time=datetime.utcnow())
        db.session.add(session)
        db.session.commit()
        return session

    def end_session(self, session_id: int) -> Optional[SessionRecord]:
        """Close a session, computing duration and aggregating counters."""
        session = SessionRecord.query.get(session_id)
        if session is None:
            return None

        session.end_time = datetime.utcnow()
        if session.start_time:
            delta = session.end_time - session.start_time
            session.duration_secs = delta.total_seconds()

        # Aggregate alert count from alert_records
        session.alert_count = AlertRecord.query.filter_by(session_id=session_id).count()

        # Aggregate detection type counts
        rows = (
            db.session.query(DetectionLog.detection_type, func.count(DetectionLog.id))
            .filter_by(session_id=session_id)
            .group_by(DetectionLog.detection_type)
            .all()
        )
        for dtype, cnt in rows:
            if dtype == 'DROWSY':
                session.drowsy_frames += cnt
            elif dtype == 'YAWN':
                session.yawn_count += cnt
            elif dtype in ('DISTRACTED', 'PHONE', 'HEAD_TURN'):
                session.distracted_frames += cnt

        db.session.commit()
        return session

    def get_session(self, session_id: int) -> Optional[SessionRecord]:
        return SessionRecord.query.get(session_id)

    def get_all_sessions(self, limit: int = 50) -> List[SessionRecord]:
        return (SessionRecord.query
                .order_by(desc(SessionRecord.start_time))
                .limit(limit)
                .all())

    # ─── Detection Logging ────────────────────────────────────────────────────

    def log_detection(self,
                      session_id: int,
                      detection_type: str,
                      ear: float = 0.0,
                      mar: float = 0.0,
                      head_pitch: float = 0.0,
                      head_yaw: float = 0.0,
                      head_roll: float = 0.0,
                      confidence: float = 1.0,
                      severity: str = 'HIGH',
                      screenshot_path: Optional[str] = None) -> DetectionLog:
        """Insert a single detection event."""
        log = DetectionLog(
            session_id=session_id,
            timestamp=datetime.utcnow(),
            detection_type=detection_type,
            ear=ear,
            mar=mar,
            head_pitch=head_pitch,
            head_yaw=head_yaw,
            head_roll=head_roll,
            confidence=confidence,
            severity=severity,
            screenshot_path=screenshot_path,
        )
        db.session.add(log)
        db.session.commit()
        return log

    def get_recent_logs(self, limit: int = 100,
                        session_id: Optional[int] = None) -> List[DetectionLog]:
        """Return the most recent detection logs, optionally filtered by session."""
        q = DetectionLog.query
        if session_id:
            q = q.filter_by(session_id=session_id)
        return q.order_by(desc(DetectionLog.timestamp)).limit(limit).all()

    def get_logs_by_type(self, detection_type: str,
                          limit: int = 50) -> List[DetectionLog]:
        return (DetectionLog.query
                .filter_by(detection_type=detection_type)
                .order_by(desc(DetectionLog.timestamp))
                .limit(limit)
                .all())

    def get_logs_by_date_range(self, start: datetime,
                                end: datetime) -> List[DetectionLog]:
        return (DetectionLog.query
                .filter(DetectionLog.timestamp >= start,
                        DetectionLog.timestamp <= end)
                .order_by(DetectionLog.timestamp)
                .all())

    def clear_logs(self) -> int:
        """Delete all detection logs and alerts. Returns rows deleted."""
        n = DetectionLog.query.delete()
        AlertRecord.query.delete()
        db.session.commit()
        return n

    # ─── Alert Logging ────────────────────────────────────────────────────────

    def log_alert(self, session_id: int, alert_type: str,
                  audio_played: bool = False,
                  tts_played: bool = False) -> AlertRecord:
        """Record that an alert was fired."""
        alert = AlertRecord(
            session_id=session_id,
            timestamp=datetime.utcnow(),
            alert_type=alert_type,
            audio_played=audio_played,
            tts_played=tts_played,
        )
        db.session.add(alert)
        db.session.commit()
        return alert

    def get_recent_alerts(self, limit: int = 50) -> List[AlertRecord]:
        return (AlertRecord.query
                .order_by(desc(AlertRecord.timestamp))
                .limit(limit)
                .all())

    # ─── Statistics & Analytics ───────────────────────────────────────────────

    def get_statistics(self) -> dict:
        """
        Compute aggregate statistics for the reports page and API.

        Returns
        -------
        dict with keys:
            total_sessions, total_detections, total_alerts,
            detection_by_type, daily_counts (last 7 days),
            hourly_distribution, avg_ear, avg_mar,
            recent_sessions
        """
        total_sessions   = SessionRecord.query.count()
        total_detections = DetectionLog.query.count()
        total_alerts     = AlertRecord.query.count()

        # Counts broken down by detection type
        type_counts = (
            db.session.query(DetectionLog.detection_type, func.count(DetectionLog.id))
            .group_by(DetectionLog.detection_type)
            .all()
        )
        detection_by_type = {dt: cnt for dt, cnt in type_counts}

        # Severity breakdown
        severity_counts = (
            db.session.query(DetectionLog.severity, func.count(DetectionLog.id))
            .group_by(DetectionLog.severity)
            .all()
        )
        severity_breakdown = {s: cnt for s, cnt in severity_counts}

        # Daily counts for the past 7 days
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        daily_rows = (
            db.session.query(
                func.date(DetectionLog.timestamp).label('day'),
                func.count(DetectionLog.id).label('cnt')
            )
            .filter(DetectionLog.timestamp >= seven_days_ago)
            .group_by(func.date(DetectionLog.timestamp))
            .all()
        )
        daily_counts = {str(row.day): row.cnt for row in daily_rows}

        # Hourly distribution (0-23)
        hourly_rows = (
            db.session.query(
                func.strftime('%H', DetectionLog.timestamp).label('hour'),
                func.count(DetectionLog.id).label('cnt')
            )
            .group_by(func.strftime('%H', DetectionLog.timestamp))
            .all()
        )
        hourly_distribution = {row.hour: row.cnt for row in hourly_rows}

        # Average EAR and MAR at time of detection
        avg_vals = db.session.query(
            func.avg(DetectionLog.ear),
            func.avg(DetectionLog.mar)
        ).first()
        avg_ear = round(avg_vals[0] or 0.0, 4)
        avg_mar = round(avg_vals[1] or 0.0, 4)

        # Recent sessions
        recent_sessions = [s.to_dict() for s in self.get_all_sessions(limit=10)]

        # Alert type breakdown
        alert_type_rows = (
            db.session.query(AlertRecord.alert_type, func.count(AlertRecord.id))
            .group_by(AlertRecord.alert_type)
            .all()
        )
        alert_by_type = {at: cnt for at, cnt in alert_type_rows}

        return {
            'total_sessions':     total_sessions,
            'total_detections':   total_detections,
            'total_alerts':       total_alerts,
            'detection_by_type':  detection_by_type,
            'severity_breakdown': severity_breakdown,
            'daily_counts':       daily_counts,
            'hourly_distribution':hourly_distribution,
            'avg_ear':            avg_ear,
            'avg_mar':            avg_mar,
            'recent_sessions':    recent_sessions,
            'alert_by_type':      alert_by_type,
        }

    def get_session_summary(self, session_id: int) -> dict:
        """Detailed summary for a single session."""
        session = self.get_session(session_id)
        if not session:
            return {}

        logs  = self.get_recent_logs(limit=1000, session_id=session_id)
        alerts = (AlertRecord.query
                  .filter_by(session_id=session_id)
                  .order_by(AlertRecord.timestamp).all())

        ear_values = [l.ear for l in logs if l.ear > 0]
        mar_values = [l.mar for l in logs if l.mar > 0]

        return {
            'session':     session.to_dict(),
            'log_count':   len(logs),
            'alert_count': len(alerts),
            'avg_ear':     round(sum(ear_values) / max(len(ear_values), 1), 4),
            'avg_mar':     round(sum(mar_values) / max(len(mar_values), 1), 4),
            'logs':        [l.to_dict() for l in logs[:50]],
            'alerts':      [a.to_dict() for a in alerts[:50]],
        }

    # ─── Driver Profiles ──────────────────────────────────────────────────────

    def get_or_create_default_profile(self) -> DriverProfile:
        profile = DriverProfile.query.filter_by(name='Default Driver').first()
        if not profile:
            profile = DriverProfile(name='Default Driver')
            db.session.add(profile)
            db.session.commit()
        return profile

    def update_profile(self, profile_id: int, **kwargs) -> Optional[DriverProfile]:
        profile = DriverProfile.query.get(profile_id)
        if not profile:
            return None
        for key, val in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, val)
        db.session.commit()
        return profile

    # ─── System Config ────────────────────────────────────────────────────────

    def get_config(self, key: str, default=None):
        row = SystemConfig.query.filter_by(key=key).first()
        return row.value if row else default

    def set_config(self, key: str, value: str, description: str = '') -> SystemConfig:
        row = SystemConfig.query.filter_by(key=key).first()
        if row:
            row.value = value
        else:
            row = SystemConfig(key=key, value=value, description=description)
            db.session.add(row)
        db.session.commit()
        return row

    def seed_default_config(self) -> None:
        """Populate SystemConfig with factory defaults if table is empty."""
        defaults = [
            ('EAR_THRESHOLD',      '0.25',  'Eye Aspect Ratio threshold for drowsiness'),
            ('MAR_THRESHOLD',      '0.6',   'Mouth Aspect Ratio threshold for yawning'),
            ('EAR_CONSEC_FRAMES',  '20',    'Consecutive frames below EAR before alert'),
            ('MAR_CONSEC_FRAMES',  '15',    'Consecutive frames above MAR before alert'),
            ('HEAD_PITCH_THRESH',  '20',    'Head pitch threshold in degrees'),
            ('HEAD_YAW_THRESH',    '30',    'Head yaw threshold in degrees'),
            ('PHONE_CONF_THRESH',  '0.5',   'YOLOv8 confidence threshold for phone detection'),
            ('ALERT_COOLDOWN',     '5',     'Seconds between repeated alerts'),
        ]
        for key, value, desc in defaults:
            if not SystemConfig.query.filter_by(key=key).first():
                db.session.add(SystemConfig(key=key, value=value, description=desc))
        db.session.commit()
