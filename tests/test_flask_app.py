"""
test_flask_app.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Integration tests for all Flask routes and API endpoints.
Run: pytest tests/test_flask_app.py -v
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import json
import tempfile

# ── App Fixture ───────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def app():
    """Create a test Flask application with an in-memory SQLite DB."""
    # Patch heavy detector initialisation before importing app
    import unittest.mock as mock
    with mock.patch('modules.face_detector.FaceDetector'),     \
         mock.patch('modules.eye_detector.EyeDetector'),        \
         mock.patch('modules.yawn_detector.YawnDetector'),      \
         mock.patch('modules.head_pose.HeadPoseEstimator'),     \
         mock.patch('modules.phone_detector.PhoneDetector'),    \
         mock.patch('modules.alert_system.AlertSystem'):

        import app as flask_app

        flask_app.app.config.update({
            'TESTING': True,
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
            'WTF_CSRF_ENABLED': False,
            'SCREENSHOT_DIR': tempfile.mkdtemp(),
            'ALERT_AUDIO_DIR': tempfile.mkdtemp(),
        })

        with flask_app.app.app_context():
            from database.models import db
            db.create_all()
            flask_app.db_manager = flask_app.DatabaseManager(flask_app.app)

        yield flask_app.app


@pytest.fixture
def client(app):
    """Create an authenticated test client using the real login route."""
    from database.models import db, User

    with app.app_context():
        user = User.query.filter_by(username="pytest_user").first()

        if user is None:
            user = User(
                username="pytest_user",
                email="pytest_user@example.com",
                full_name="Pytest User",
                role="admin"
            )
            user.set_password("pytest_password")
            db.session.add(user)
            db.session.commit()

    test_client = app.test_client()

    response = test_client.post(
        "/login",
        data={
            "username": "pytest_user",
            "password": "pytest_password"
        },
        follow_redirects=True
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Test login failed: HTTP {response.status_code}"
        )

    return test_client


# ── Page Route Tests ──────────────────────────────────────────────────────────

class TestPageRoutes:
    def test_index_returns_200(self, client):
        r = client.get('/')
        assert r.status_code == 200

    def test_dashboard_returns_200(self, client):
        r = client.get('/dashboard')
        assert r.status_code == 200

    def test_logs_returns_200(self, client):
        r = client.get('/logs')
        assert r.status_code == 200

    def test_reports_returns_200(self, client):
        r = client.get('/reports')
        assert r.status_code == 200

    def test_settings_returns_200(self, client):
        r = client.get('/settings')
        assert r.status_code == 200

    def test_index_contains_brand(self, client):
        r = client.get('/')
        assert b'DrowsyGuard' in r.data or b'Driver' in r.data

    def test_dashboard_contains_video_element(self, client):
        r = client.get('/dashboard')
        assert b'videoFeed' in r.data or b'video' in r.data.lower()


# ── API Endpoint Tests ────────────────────────────────────────────────────────

class TestAPIEndpoints:
    def test_api_stats_returns_json(self, client):
        r = client.get('/api/stats')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert 'total_sessions' in data
        assert 'total_detections' in data
        assert 'total_alerts' in data

    def test_api_logs_returns_list(self, client):
        r = client.get('/api/logs')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert isinstance(data, list)

    def test_api_driver_state_returns_json(self, client):
        r = client.get('/api/driver_state')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert 'status' in data

    def test_api_config_get(self, client):
        r = client.get('/api/config')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert 'EAR_THRESHOLD' in data

    def test_api_config_post_valid(self, client):
        payload = {'EAR_THRESHOLD': 0.28}
        r = client.post('/api/config',
                        data=json.dumps(payload),
                        content_type='application/json')
        assert r.status_code == 200
        resp = json.loads(r.data)
        assert resp['status'] == 'updated'

    def test_api_screenshots_returns_dict(self, client):
        r = client.get('/api/screenshots')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert 'screenshots' in data
        assert isinstance(data['screenshots'], list)

    def test_api_logs_limit_param(self, client):
        r = client.get('/api/logs?limit=5')
        assert r.status_code == 200


# ── Detection Control Tests ───────────────────────────────────────────────────

class TestDetectionControl:
    def test_start_detection(self, client):
        r = client.post('/start_detection')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert data['status'] in ('started', 'already_running')

    def test_stop_detection(self, client):
        r = client.post('/stop_detection')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert data['status'] == 'stopped'

    def test_clear_logs(self, client):
        r = client.post('/api/clear_logs')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert data['status'] == 'cleared'


# ── Database Integration Tests ────────────────────────────────────────────────

class TestDatabase:
    def test_create_session(self, app):
        with app.app_context():
            from database.db_manager import DatabaseManager
            mgr = DatabaseManager(app)
            sess = mgr.create_session()
            assert sess.id is not None
            assert sess.start_time is not None

    def test_log_detection(self, app):
        with app.app_context():
            from database.db_manager import DatabaseManager
            mgr  = DatabaseManager(app)
            sess = mgr.create_session()
            log  = mgr.log_detection(
                session_id=sess.id,
                detection_type='DROWSY',
                ear=0.18, mar=0.25,
                head_pitch=5.0, head_yaw=3.0
            )
            assert log.id is not None
            assert log.detection_type == 'DROWSY'

    def test_get_recent_logs(self, app):
        with app.app_context():
            from database.db_manager import DatabaseManager
            mgr  = DatabaseManager(app)
            logs = mgr.get_recent_logs(limit=10)
            assert isinstance(logs, list)

    def test_get_statistics(self, app):
        with app.app_context():
            from database.db_manager import DatabaseManager
            mgr   = DatabaseManager(app)
            stats = mgr.get_statistics()
            assert 'total_sessions'   in stats
            assert 'total_detections' in stats
            assert 'detection_by_type' in stats

    def test_end_session(self, app):
        with app.app_context():
            from database.db_manager import DatabaseManager
            mgr  = DatabaseManager(app)
            sess = mgr.create_session()
            ended = mgr.end_session(sess.id)
            assert ended.end_time is not None

    def test_log_alert(self, app):
        with app.app_context():
            from database.db_manager import DatabaseManager
            mgr  = DatabaseManager(app)
            sess = mgr.create_session()
            alert = mgr.log_alert(sess.id, 'DROWSY', audio_played=True)
            assert alert.id is not None
            assert alert.alert_type == 'DROWSY'
