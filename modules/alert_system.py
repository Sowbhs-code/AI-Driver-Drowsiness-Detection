"""
alert_system.py
─────────────────────────────────────────────────────────
Multi-channel alert system for the drowsiness detection
system.

Channels supported
──────────────────
  1. Audio file playback  (pygame / playsound)
  2. Text-to-Speech       (pyttsx3)
  3. Visual overlay       (returned as flag; drawn in app.py)
  4. Cooldown management  (prevents alert spam)

Alert types
───────────
  DROWSY      – driver eyes closed too long
  YAWN        – persistent yawning detected
  DISTRACTED  – head turned / phone in hand / face missing
  PHONE       – mobile phone detected

Usage
─────
    alert = AlertSystem(audio_dir='alerts/')
    alert.trigger('DROWSY')
─────────────────────────────────────────────────────────
"""

import os
import time
import threading
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Optional audio backends ────────────────────────────────────────────────────
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False
    logger.warning("pygame unavailable – falling back to pyttsx3 TTS only.")

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False
    logger.warning("pyttsx3 unavailable – TTS alerts disabled.")


# ─── Alert Configuration ──────────────────────────────────────────────────────

ALERT_CONFIG: Dict[str, dict] = {
    'DROWSY': {
        'audio_file':  'alert_drowsy.mp3',
        'tts_message': 'Warning! Drowsiness detected. Please pull over and rest.',
        'priority':    3,           # higher = more urgent
        'cooldown':    5.0,         # seconds between repeated alerts
        'color':       (0, 0, 255), # BGR – red overlay
    },
    'YAWN': {
        'audio_file':  'alert_yawn.mp3',
        'tts_message': 'Yawning detected. You may be getting tired.',
        'priority':    2,
        'cooldown':    8.0,
        'color':       (0, 165, 255),
    },
    'DISTRACTED': {
        'audio_file':  'alert_distracted.mp3',
        'tts_message': 'Distraction detected. Please focus on the road.',
        'priority':    2,
        'cooldown':    5.0,
        'color':       (0, 165, 255),
    },
    'PHONE': {
        'audio_file':  'alert_phone.mp3',
        'tts_message': 'Phone usage detected while driving. Please put down your phone.',
        'priority':    3,
        'cooldown':    6.0,
        'color':       (0, 0, 255),
    },
    'NO_FACE': {
        'audio_file':  'alert_distracted.mp3',
        'tts_message': 'Driver face not detected. Are you looking at the road?',
        'priority':    1,
        'cooldown':    10.0,
        'color':       (0, 200, 255),
    },
}


class AlertSystem:
    """
    Thread-safe multi-channel alert manager with cooldown control.

    Parameters
    ----------
    audio_dir       : str   folder containing .mp3/.wav alert files
    cooldown_secs   : float global minimum gap between any two alerts
    tts_enabled     : bool  enable text-to-speech fallback
    audio_enabled   : bool  enable pygame audio playback
    volume          : float 0.0–1.0 playback volume
    """

    def __init__(self,
                 audio_dir: str = 'alerts',
                 cooldown_secs: float = 5.0,
                 tts_enabled: bool = True,
                 audio_enabled: bool = True,
                 volume: float = 0.9):

        self.audio_dir      = audio_dir
        self.cooldown_secs  = cooldown_secs
        self.tts_enabled    = tts_enabled and PYTTSX3_AVAILABLE
        self.audio_enabled  = audio_enabled and PYGAME_AVAILABLE
        self.volume         = volume

        # Timestamps of last alert per type  {alert_type: float}
        self._last_alert:   Dict[str, float] = {}
        self._global_last:  float = 0.0

        # TTS engine (initialised lazily to avoid thread issues)
        self._tts_engine    = None
        self._tts_lock      = threading.Lock()

        # Audio cache  {filename: pygame.mixer.Sound}
        self._audio_cache:  Dict[str, object] = {}

        # Statistics
        self._alert_counts: Dict[str, int] = {k: 0 for k in ALERT_CONFIG}
        self._total_alerts  = 0

        self._generate_placeholder_audio_files()

    # ── Public API ────────────────────────────────────────────────────────────

    def trigger(self, alert_type: str,
                force: bool = False) -> dict:
        """
        Fire an alert of the given type.

        Parameters
        ----------
        alert_type : str   one of DROWSY | YAWN | DISTRACTED | PHONE | NO_FACE
        force      : bool  bypass cooldown (e.g. manual test)

        Returns
        -------
        dict
            fired         : bool  – whether the alert actually played
            alert_type    : str
            channel       : str   – 'audio' | 'tts' | 'none'
            cooldown_remaining : float
        """
        cfg = ALERT_CONFIG.get(alert_type)
        if cfg is None:
            logger.warning(f"Unknown alert type: {alert_type}")
            return {'fired': False, 'alert_type': alert_type,
                    'channel': 'none', 'cooldown_remaining': 0.0}

        now = time.time()

        # Per-type cooldown check
        last   = self._last_alert.get(alert_type, 0.0)
        type_cooldown = cfg['cooldown']
        remaining = type_cooldown - (now - last)

        if not force and remaining > 0:
            return {'fired': False, 'alert_type': alert_type,
                    'channel': 'none', 'cooldown_remaining': round(remaining, 1)}

        # Global cooldown (prevent overlapping audio)
        global_remaining = self.cooldown_secs - (now - self._global_last)
        if not force and global_remaining > 0 and alert_type not in ('DROWSY', 'PHONE'):
            return {'fired': False, 'alert_type': alert_type,
                    'channel': 'none', 'cooldown_remaining': round(global_remaining, 1)}

        # Update timestamps
        self._last_alert[alert_type] = now
        self._global_last = now
        self._alert_counts[alert_type] = self._alert_counts.get(alert_type, 0) + 1
        self._total_alerts += 1

        # Fire in background thread so video stream is not blocked
        channel = self._play_alert_async(alert_type, cfg)

        logger.info(f"[ALERT] {alert_type} fired via {channel}")
        return {'fired': True, 'alert_type': alert_type,
                'channel': channel, 'cooldown_remaining': 0.0}

    def test_alert(self, alert_type: str = 'DROWSY') -> dict:
        """Force-fire an alert ignoring cooldown (used in settings page)."""
        return self.trigger(alert_type, force=True)

    def get_stats(self) -> dict:
        return {
            'total_alerts':  self._total_alerts,
            'by_type':       dict(self._alert_counts),
            'audio_enabled': self.audio_enabled,
            'tts_enabled':   self.tts_enabled,
        }

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, volume))
        if PYGAME_AVAILABLE:
            pygame.mixer.music.set_volume(self.volume)

    def reset_cooldowns(self) -> None:
        """Clear all cooldown timers (call between sessions)."""
        self._last_alert.clear()
        self._global_last = 0.0

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _play_alert_async(self, alert_type: str, cfg: dict) -> str:
        """Spawn a daemon thread to play audio or TTS without blocking."""
        thread = threading.Thread(
            target=self._play_alert,
            args=(alert_type, cfg),
            daemon=True
        )
        thread.start()

        if self.audio_enabled:
            return 'audio'
        elif self.tts_enabled:
            return 'tts'
        return 'visual_only'

    def _play_alert(self, alert_type: str, cfg: dict) -> None:
        """Actual playback — runs in worker thread."""
        played = False

        # Try pygame audio first
        if self.audio_enabled:
            played = self._play_audio_file(cfg['audio_file'])

        # Fallback to TTS
        if not played and self.tts_enabled:
            self._speak(cfg['tts_message'])

    def _play_audio_file(self, filename: str) -> bool:
        """Play an audio file using pygame.mixer. Returns True on success."""
        if not PYGAME_AVAILABLE:
            return False

        filepath = os.path.join(self.audio_dir, filename)
        if not os.path.exists(filepath):
            # Try .wav variant
            wav_path = filepath.replace('.mp3', '.wav')
            if os.path.exists(wav_path):
                filepath = wav_path
            else:
                return False

        try:
            if filepath not in self._audio_cache:
                self._audio_cache[filepath] = pygame.mixer.Sound(filepath)
            sound = self._audio_cache[filepath]
            sound.set_volume(self.volume)
            sound.play()
            return True
        except Exception as e:
            logger.error(f"Audio playback error ({filename}): {e}")
            return False

    def _speak(self, text: str) -> None:
        """Speak text using pyttsx3 TTS engine."""
        if not PYTTSX3_AVAILABLE:
            return
        with self._tts_lock:
            try:
                if self._tts_engine is None:
                    self._tts_engine = pyttsx3.init()
                    self._tts_engine.setProperty('rate', 175)
                    self._tts_engine.setProperty('volume', self.volume)
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS error: {e}")
                self._tts_engine = None  # reset engine on failure

    def _generate_placeholder_audio_files(self) -> None:
        """
        Write a README into the alerts/ directory explaining which
        audio files are expected.  The system works without them via TTS.
        """
        os.makedirs(self.audio_dir, exist_ok=True)
        readme = os.path.join(self.audio_dir, 'README.txt')
        if not os.path.exists(readme):
            content = (
                "ALERT AUDIO FILES\n"
                "═════════════════\n"
                "Place the following audio files in this directory:\n\n"
                "  alert_drowsy.mp3      – plays when drowsiness is detected\n"
                "  alert_yawn.mp3        – plays when yawning is detected\n"
                "  alert_distracted.mp3  – plays when distraction is detected\n"
                "  alert_phone.mp3       – plays when phone usage is detected\n\n"
                "If files are missing, the system falls back to pyttsx3 TTS.\n"
                "Recommended: short beep/voice clips (1–3 seconds).\n"
                "Free alert sounds: https://freesound.org\n"
            )
            with open(readme, 'w') as f:
                f.write(content)

        # Generate minimal WAV beep files using Python stdlib (no deps needed)
        self._generate_beep_wav('alert_drowsy.wav',     frequency=880, duration_ms=800)
        self._generate_beep_wav('alert_yawn.wav',       frequency=660, duration_ms=600)
        self._generate_beep_wav('alert_distracted.wav', frequency=760, duration_ms=700)
        self._generate_beep_wav('alert_phone.wav',      frequency=880, duration_ms=800)

    def _generate_beep_wav(self, filename: str,
                            frequency: int = 880,
                            duration_ms: int = 500,
                            sample_rate: int = 44100,
                            amplitude: int = 16000) -> None:
        """Generate a simple sine-wave beep WAV file using only stdlib."""
        import struct
        import math
        import wave

        filepath = os.path.join(self.audio_dir, filename)
        if os.path.exists(filepath):
            return  # don't overwrite existing files

        num_samples = int(sample_rate * duration_ms / 1000)
        samples = []
        for i in range(num_samples):
            # Sine wave with fade-in / fade-out envelope
            t = i / sample_rate
            envelope = 1.0
            fade = int(sample_rate * 0.05)   # 50 ms fade
            if i < fade:
                envelope = i / fade
            elif i > num_samples - fade:
                envelope = (num_samples - i) / fade
            val = int(amplitude * envelope * math.sin(2 * math.pi * frequency * t))
            samples.append(struct.pack('<h', max(-32767, min(32767, val))))

        try:
            with wave.open(filepath, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(b''.join(samples))
            logger.debug(f"Generated beep WAV: {filepath}")
        except Exception as e:
            logger.warning(f"Could not generate WAV {filename}: {e}")


# ─── Convenience Singleton Factory ────────────────────────────────────────────

_instance: Optional[AlertSystem] = None


def get_alert_system(audio_dir: str = 'alerts') -> AlertSystem:
    """Return a module-level singleton AlertSystem instance."""
    global _instance
    if _instance is None:
        _instance = AlertSystem(audio_dir=audio_dir)
    return _instance
