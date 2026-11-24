"""
test_alert_system.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Unit tests for the AlertSystem module.
Run: pytest tests/test_alert_system.py -v
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys, os, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from modules.alert_system import AlertSystem, ALERT_CONFIG


@pytest.fixture
def alert(tmp_path):
    return AlertSystem(
        audio_dir=str(tmp_path),
        cooldown_secs=1.0,
        tts_enabled=False,
        audio_enabled=False,  # disable audio in tests
    )


class TestAlertSystem:
    def test_instantiation(self, alert):
        assert alert is not None

    def test_trigger_returns_dict(self, alert):
        result = alert.trigger('DROWSY', force=True)
        assert isinstance(result, dict)
        assert 'fired' in result
        assert 'alert_type' in result

    def test_trigger_force_fires(self, alert):
        result = alert.trigger('DROWSY', force=True)
        assert result['fired'] is True

    def test_cooldown_prevents_repeat(self, alert):
        alert.trigger('YAWN', force=True)
        result = alert.trigger('YAWN')          # immediate repeat — should be blocked
        assert result['fired'] is False
        assert result['cooldown_remaining'] > 0

    def test_cooldown_expires(self, alert):
        alert_short = AlertSystem(
            audio_dir=str(tempfile.mkdtemp()),
            cooldown_secs=0.1,
            tts_enabled=False,
            audio_enabled=False,
        )
        alert_short.trigger('PHONE', force=True)
        time.sleep(0.15)
        # Patch per-type cooldown to be short too
        from modules.alert_system import ALERT_CONFIG
        original = ALERT_CONFIG['PHONE']['cooldown']
        ALERT_CONFIG['PHONE']['cooldown'] = 0.05
        result = alert_short.trigger('PHONE')
        ALERT_CONFIG['PHONE']['cooldown'] = original
        assert result['fired'] is True

    def test_unknown_type_returns_not_fired(self, alert):
        result = alert.trigger('UNKNOWN_TYPE')
        assert result['fired'] is False

    def test_get_stats(self, alert):
        alert.trigger('DROWSY', force=True)
        stats = alert.get_stats()
        assert 'total_alerts' in stats
        assert stats['total_alerts'] >= 1

    def test_all_alert_types_trigger(self, alert):
        for atype in ('DROWSY', 'YAWN', 'DISTRACTED', 'PHONE', 'NO_FACE'):
            r = alert.trigger(atype, force=True)
            assert r['alert_type'] == atype

    def test_set_volume(self, alert):
        alert.set_volume(0.5)
        assert alert.volume == 0.5

    def test_volume_clamp(self, alert):
        alert.set_volume(2.0)
        assert alert.volume <= 1.0
        alert.set_volume(-1.0)
        assert alert.volume >= 0.0

    def test_reset_cooldowns(self, alert):
        alert.trigger('DROWSY', force=True)
        alert.reset_cooldowns()
        result = alert.trigger('DROWSY')
        assert result['fired'] is True

    def test_wav_files_created(self, tmp_path):
        AlertSystem(audio_dir=str(tmp_path), tts_enabled=False, audio_enabled=False)
        wav_files = list(tmp_path.glob('*.wav'))
        assert len(wav_files) > 0

    def test_readme_created(self, tmp_path):
        AlertSystem(audio_dir=str(tmp_path), tts_enabled=False, audio_enabled=False)
        assert (tmp_path / 'README.txt').exists()

    def test_alert_config_completeness(self):
        required_keys = {'audio_file', 'tts_message', 'priority', 'cooldown', 'color'}
        for atype, cfg in ALERT_CONFIG.items():
            for key in required_keys:
                assert key in cfg, f"ALERT_CONFIG['{atype}'] missing key '{key}'"
