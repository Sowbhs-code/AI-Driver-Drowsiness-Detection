/**
 * main.js — Global utilities shared across all pages
 * DrowsyGuard AI Driver Monitoring System
 */
'use strict';

// Live Clock
(function () {
  const el = document.getElementById('topbarClock');
  if (!el) return;
  const tick = () => { el.textContent = new Date().toLocaleTimeString('en-IN', { hour12: false }); };
  tick(); setInterval(tick, 1000);
})();

// Sidebar Toggle (mobile)
(function () {
  const btn = document.getElementById('sidebarToggle');
  const sb  = document.getElementById('sidebar');
  if (!btn || !sb) return;
  btn.addEventListener('click', () => sb.classList.toggle('open'));
  document.addEventListener('click', e => {
    if (window.innerWidth <= 768 && !sb.contains(e.target) && !btn.contains(e.target))
      sb.classList.remove('open');
  });
})();

// Socket.IO connection status
function initSocket() {
  if (typeof io === 'undefined') { setConn(false); return null; }
  try {
    const s = io({ transports: ['websocket', 'polling'] });
    s.on('connect',       () => setConn(true));
    s.on('disconnect',    () => setConn(false));
    s.on('connect_error', () => setConn(false));
    return s;
  } catch (e) { setConn(false); return null; }
}

function setConn(ok) {
  const dot = document.getElementById('connectionDot');
  const lbl = document.getElementById('connectionLabel');
  if (!dot) return;
  dot.classList.toggle('connected', ok);
  dot.classList.toggle('disconnected', !ok);
  if (lbl) lbl.textContent = ok ? 'Connected' : 'Disconnected';
}

window.drowsySocket = initSocket();

// Toast notifications
let _toastTimer = null;
function showToast(msg, type = 'error') {
  const toast = document.getElementById('alertToast');
  const span  = document.getElementById('toastMessage');
  if (!toast) return;
  const c = { error: '#ef4444', warning: '#f59e0b', info: '#3b82f6', success: '#10b981' };
  toast.style.background = c[type] || c.error;
  if (span) span.textContent = msg;
  toast.classList.add('show');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => toast.classList.remove('show'), 4000);
}

// API helpers
async function apiGet(url) {
  try { const r = await fetch(url); return r.ok ? r.json() : null; }
  catch { return null; }
}
async function apiPost(url, data = {}) {
  try {
    const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
    return r.ok ? r.json() : null;
  } catch { return null; }
}

function formatDuration(s) {
  return [Math.floor(s/3600), Math.floor((s%3600)/60), Math.floor(s%60)]
    .map(v => String(v).padStart(2,'0')).join(':');
}

function updateSliderVal(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

window.showToast      = showToast;
window.apiGet         = apiGet;
window.apiPost        = apiPost;
window.formatDuration = formatDuration;
window.updateSliderVal = updateSliderVal;
