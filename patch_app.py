import shutil

shutil.copy("app.py", "app_before_patch.py")          # safety copy
lines = open("app.py", encoding="utf-8").read().split("\n")

SG_BLOCK = [
    "if CONFIG['USE_CNN_SUNGLASSES'] and cnn_models.sunglasses_ready:",
    "    if frame_count % int(CONFIG['CNN_SG_INTERVAL']) == 0:",
    "        p_sg = cnn_models.sunglasses_prob(frame, landmarks)",
    "        if p_sg is not None:",
    "            if p_sg > CONFIG['CNN_SG_THRESHOLD']:",
    "                sg_pos += 1; sg_neg = 0",
    "            else:",
    "                sg_neg += 1; sg_pos = 0",
    "            if sg_pos >= CONFIG['CNN_SG_CONFIRM']:",
    "                sg_state = True",
    "            elif sg_neg >= CONFIG['CNN_SG_RELEASE']:",
    "                sg_state = False",
    "    wearing_sunglasses = sg_state",
]

EYE_BLOCK = [
    "eye_closed = ear < CONFIG['EAR_THRESHOLD']",
    "if eye_closed and CONFIG['USE_CNN_EYE'] and cnn_models.eye_ready:",
    "    p_open = cnn_models.eye_open_prob(frame, landmarks)",
    "    if p_open is not None and p_open >= CONFIG['CNN_EYE_OPEN_MIN']:",
    "        eye_closed = False",
    "if eye_closed:",
    "    drowsy_counter += 1",
    "    driver_state['eye_status'] = 'Closed'",
    "else:",
    "    drowsy_counter = 0",
    "    driver_state['eye_status'] = 'Open'",
]

OLD_EYE = ["if ear < CONFIG['EAR_THRESHOLD']:", "drowsy_counter += 1",
           "driver_state['eye_status'] = 'Closed'", "else:",
           "drowsy_counter = 0", "driver_state['eye_status'] = 'Open'"]

out, done, i = [], set(), 0
while i < len(lines):
    line = lines[i]
    s = line.strip()
    ind = line[:len(line) - len(line.lstrip())]
    if s.startswith("head_nod_counter = 0") and "fallback" in s and "counters" not in done:
        out += [line, ind + "sg_state = False", ind + "sg_pos = 0", ind + "sg_neg = 0"]
        done.add("counters")
    elif s == "wearing_sunglasses = sg_data['sunglasses_detected']" and "sunglasses" not in done:
        out += [line] + [ind + b for b in SG_BLOCK]
        done.add("sunglasses")
    elif (s == OLD_EYE[0] and "eye" not in done
          and [l.strip() for l in lines[i:i + 6]] == OLD_EYE):
        out += [ind + b for b in EYE_BLOCK]
        done.add("eye")
        i += 6
        continue
    else:
        out.append(line)
    i += 1

print("patched:", sorted(done))
missing = {"counters", "sunglasses", "eye"} - done
if missing:
    print("NOT patched (nothing was written):", sorted(missing))
else:
    open("app.py", "w", encoding="utf-8").write("\n".join(out))
    print("app.py updated. Backup saved as app_before_patch.py")