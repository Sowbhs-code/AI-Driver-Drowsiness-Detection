import shutil, sys
OLD = """            if mar > CONFIG['MAR_THRESHOLD']:
                yawn_counter += 1
                driver_state['yawn_status'] = 'Yawning'
            else:
                yawn_counter = 0
                driver_state['yawn_status'] = 'Normal'
"""
NEW = """            if mar > CONFIG['MAR_THRESHOLD']:
                yawn_counter += 1
            else:
                yawn_counter = 0
            # show 'Yawning' only when the mouth stays wide open (not while talking)
            driver_state['yawn_status'] = ('Yawning' if yawn_counter >= CONFIG['MAR_CONSEC_FRAMES']
                                           else 'Normal')
"""
src = open("app.py", encoding="utf-8").read()
if src.count(OLD) != 1:
    sys.exit("Could not find the exact yawn lines in app.py (found %d). Nothing changed." % src.count(OLD))
shutil.copy("app.py", "app_before_yawnfix.py")
open("app.py", "w", encoding="utf-8").write(src.replace(OLD, NEW))
print("patched. Backup saved as app_before_yawnfix.py")