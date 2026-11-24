import shutil

shutil.copy("app.py", "app_before_distraction.py")      # safety copy
src = open("app.py", encoding="utf-8").read()

edits = [
    # 1. only the eye-based check can raise DROWSY
    ("if drowsy_by_ear or drowsy_by_head_nod:",
     "if drowsy_by_ear:"),
    # 2. the sunglasses head-droop now raises DISTRACTED
    ("elif distracted_counter >= CONFIG['DISTRACTED_FRAMES'] or driver_state['phone_detected']:",
     "elif drowsy_by_head_nod or distracted_counter >= CONFIG['DISTRACTED_FRAMES'] or driver_state['phone_detected']:"),
    # 3. message on the video
    ('_draw_alert_overlay(display_frame, "DISTRACTION ALERT!", (0, 165, 255))',
     '_draw_alert_overlay(display_frame, "DISTRACTION ALERT! (head down, sunglasses on)" '
     'if drowsy_by_head_nod else "DISTRACTION ALERT!", (0, 165, 255))'),
    # 4. reset the head-droop counter after the alert, so it does not repeat every frame
    ("            elif alert_type == 'DISTRACTED':\n                distracted_counter = 0",
     "            elif alert_type == 'DISTRACTED':\n                distracted_counter = 0\n                head_nod_counter = 0"),
]

for old, new in edits:
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"STOPPED: expected 1 match, found {n} for:\n{old}\n(app.py was not changed)")
    src = src.replace(old, new)

open("app.py", "w", encoding="utf-8").write(src)
print("app.py updated. Backup saved as app_before_distraction.py")