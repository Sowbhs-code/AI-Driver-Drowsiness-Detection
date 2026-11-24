"""
conftest.py — Shared pytest fixtures and configuration.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np


@pytest.fixture(scope='session')
def sample_bgr_frame():
    """A realistic-looking 480×640 BGR frame with a light face region."""
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    # Simulate a skin-tone ellipse (face placeholder)
    import cv2
    cv2.ellipse(frame, (320, 240), (100, 130), 0, 0, 360, (180, 140, 110), -1)
    return frame


@pytest.fixture(scope='session')
def fake_landmarks_468():
    """Generate reproducible fake 468-point MediaPipe landmarks."""
    np.random.seed(42)
    return [(np.random.randint(100, 540), np.random.randint(80, 400),
             float(np.random.uniform(-0.1, 0.1))) for _ in range(468)]


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "unit: marks unit tests")
