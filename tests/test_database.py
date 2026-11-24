"""
test_database.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Unit tests for database models and db_manager.
Run: pytest tests/test_database.py -v
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime


@pytest.fixture(scope='module')
def flask_app():
    import unittest.mock as mock
    with mock.patch('modules.face_detector.FaceDetector'), \
         mock.patch('modules.eye_detector.EyeDetector'),   \
         mock.patch('modules.yawn_detector.YawnDetector'), \
         mock.patch('modules.head_pose.HeadPoseEstimator'),\
         mock.patch('modules.phone_detector.PhoneDetector'),\
         mock.patch('modules.alert_system.AlertSystem'):
        import app as flask_app
        flask_app.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        flask_app.app.config['TESTING'] = True
        with flask_app.app.app_context():
            from database.models import db
            db.create_all()
        yield flask_app.app


@pytest.fixture
def ctx(flask_app):
    with flask_app.app_context():
        yield flask_app


class TestOrmModels:
    def test_session_record_creation(self, ctx):
        from database.models import db, SessionRecord
        s = SessionRecord(start_time=datetime.utcnow())
        db.session.add(s); db.session.commit()
        assert s.id is not None

    def test_session_to_dict(self, ctx):
        from database.models import db, SessionRecord
        s = SessionRecord(); db.session.add(s); db.session.commit()
        d = s.to_dict()
        assert 'id' in d and 'alert_count' in d

    def test_detection_log_creation(self, ctx):
        from database.models import db, SessionRecord, DetectionLog
        s = SessionRecord(); db.session.add(s); db.session.commit()
        log = DetectionLog(session_id=s.id, detection_type='DROWSY', ear=0.20, mar=0.30)
        db.session.add(log); db.session.commit()
        assert log.id is not None

    def test_detection_log_to_dict(self, ctx):
        from database.models import db, SessionRecord, DetectionLog
        s = SessionRecord(); db.session.add(s); db.session.commit()
        log = DetectionLog(session_id=s.id, detection_type='YAWN', ear=0.30, mar=0.65)
        db.session.add(log); db.session.commit()
        d = log.to_dict()
        assert d['detection_type'] == 'YAWN'
        assert d['ear'] == 0.30

    def test_alert_record_creation(self, ctx):
        from database.models import db, SessionRecord, AlertRecord
        s = SessionRecord(); db.session.add(s); db.session.commit()
        a = AlertRecord(session_id=s.id, alert_type='PHONE', audio_played=True)
        db.session.add(a); db.session.commit()
        assert a.id is not None
        assert a.audio_played is True

    def test_driver_profile_defaults(self, ctx):
        from database.models import db, DriverProfile
        p = DriverProfile(name='Test Driver')
        db.session.add(p); db.session.commit()
        assert p.is_active is True
        assert p.ear_threshold is None  # uses global default


class TestDbManager:
    def test_full_session_lifecycle(self, ctx):
        from database.db_manager import DatabaseManager
        mgr  = DatabaseManager()
        sess = mgr.create_session()
        assert sess.id is not None
        mgr.log_detection(sess.id, 'DROWSY', ear=0.18, mar=0.25)
        mgr.log_alert(sess.id, 'DROWSY', audio_played=True)
        ended = mgr.end_session(sess.id)
        assert ended.end_time is not None
        assert ended.duration_secs >= 0

    def test_statistics_structure(self, ctx):
        from database.db_manager import DatabaseManager
        mgr   = DatabaseManager()
        stats = mgr.get_statistics()
        expected = ['total_sessions','total_detections','total_alerts',
                    'detection_by_type','daily_counts','avg_ear','avg_mar']
        for k in expected:
            assert k in stats, f"Missing stat key: {k}"

    def test_clear_logs(self, ctx):
        from database.db_manager import DatabaseManager
        mgr  = DatabaseManager()
        sess = mgr.create_session()
        mgr.log_detection(sess.id, 'YAWN')
        mgr.clear_logs()
        logs = mgr.get_recent_logs(limit=100)
        assert len(logs) == 0

    def test_config_set_get(self, ctx):
        from database.db_manager import DatabaseManager
        mgr = DatabaseManager()
        mgr.set_config('TEST_KEY', '42', 'test config')
        val = mgr.get_config('TEST_KEY')
        assert val == '42'

    def test_config_missing_key_returns_default(self, ctx):
        from database.db_manager import DatabaseManager
        mgr = DatabaseManager()
        val = mgr.get_config('NONEXISTENT_KEY', default='fallback')
        assert val == 'fallback'
