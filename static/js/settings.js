'use strict';

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('thresholdForm')?.addEventListener('submit', saveThresholds);

  // Load saved alert settings and sync toggle states
  loadAlertSettings();

  // Wire toggle switches to persist on change
  ['audioEnabled', 'ttsEnabled', 'screenshotEnabled'].forEach(id => {
    document.getElementById(id)?.addEventListener('change', saveAlertSettings);
  });
});

/* ── Threshold Form ─────────────────────────────────────────────────────── */

async function saveThresholds(e) {
  e.preventDefault();
  const form = document.getElementById('thresholdForm');
  const data = {};
  form.querySelectorAll('input[type=range]').forEach(inp => {
    data[inp.name] = parseFloat(inp.value);
  });
  const r = await apiPost('/api/config', data);
  showToast(r ? 'Thresholds saved successfully' : 'Failed to save thresholds',
            r ? 'success' : 'error');
}

function resetDefaults() {
  const defaults = {
    EAR_THRESHOLD: 0.25,
    MAR_THRESHOLD: 0.6,
    EAR_CONSEC_FRAMES: 20,
    HEAD_PITCH_THRESHOLD: 20,
    HEAD_YAW_THRESHOLD: 30,
    PHONE_CONF_THRESHOLD: 0.5,
  };
  const sliderMap = {
    EAR_THRESHOLD:      ['earThreshold',   'earThreshVal'],
    MAR_THRESHOLD:      ['marThreshold',   'marThreshVal'],
    EAR_CONSEC_FRAMES:  ['earFrames',      'earFramesVal'],
    HEAD_PITCH_THRESHOLD: ['headPitch',    'headPitchVal'],
    HEAD_YAW_THRESHOLD: ['headYaw',        'headYawVal'],
    PHONE_CONF_THRESHOLD: ['phoneConf',    'phoneConfVal'],
  };
  Object.entries(defaults).forEach(([key, val]) => {
    const [inputId, labelId] = sliderMap[key] || [];
    if (!inputId) return;
    const inp = document.getElementById(inputId);
    if (inp) inp.value = val;
    updateSliderVal(labelId, val);
  });
  showToast('Reset to defaults', 'info');
}

/* ── Alert Settings Toggles ─────────────────────────────────────────────── */

async function loadAlertSettings() {
  try {
    const r = await fetch('/api/alert_settings');
    if (!r.ok) return;
    const data = await r.json();
    const audio = document.getElementById('audioEnabled');
    const tts   = document.getElementById('ttsEnabled');
    const ss    = document.getElementById('screenshotEnabled');
    if (audio) audio.checked = !!data.audio_enabled;
    if (tts)   tts.checked   = !!data.tts_enabled;
    if (ss)    ss.checked    = !!data.screenshot_enabled;
  } catch (_) {}
}

async function saveAlertSettings() {
  const audio = document.getElementById('audioEnabled');
  const tts   = document.getElementById('ttsEnabled');
  const ss    = document.getElementById('screenshotEnabled');
  const payload = {
    audio_enabled:      audio ? audio.checked : true,
    tts_enabled:        tts   ? tts.checked   : true,
    screenshot_enabled: ss    ? ss.checked    : true,
  };
  const r = await apiPost('/api/alert_settings', payload);
  showToast(r ? 'Alert settings saved' : 'Failed to save alert settings',
            r ? 'success' : 'error');
}

/* ── Test Alerts ─────────────────────────────────────────────────────────── */

async function testAlert(type) {
  const r = await apiPost('/api/test_alert', { type });
  if (r && r.status === 'ok') {
    const ch = r.result?.channel || 'none';
    showToast(`Test alert: ${type} (channel: ${ch})`, 'warning');
  } else {
    showToast(`Failed to trigger test alert: ${type}`, 'error');
  }
}

/* ── Danger Zone ─────────────────────────────────────────────────────────── */

async function confirmClearLogs() {
  if (!confirm('Permanently delete all detection logs and alerts?')) return;
  const r = await apiPost('/api/clear_logs');
  showToast(r ? 'All logs cleared' : 'Failed to clear logs',
            r ? 'success' : 'error');
  if (r) setTimeout(() => location.reload(), 800);
}

window.resetDefaults    = resetDefaults;
window.testAlert        = testAlert;
window.confirmClearLogs = confirmClearLogs;
window.saveAlertSettings = saveAlertSettings;
