# TRAINING_GUIDE - from empty folders to real results (Windows 11, VS Code)

Run every command from the project folder in the VS Code terminal.

## 0. Setup (once)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
(Use Python 3.10-3.12. Do not install dlib - it is not needed.)

## 1. Collect REAL data
| What | Where | Put it in |
|---|---|---|
| Open/closed eyes | MRL Eye Dataset: https://mrl.cs.vsb.cz/eyedataset  (or Kaggle "drowsiness dataset") | `dataset/eye_dataset/open` and `/closed` |
| Phone (optional) | Roboflow Universe, search "cell phone" (YOLOv8 export) | `dataset/phone_data` |
| Sunglasses / yawn / test faces | Kaggle (search "sunglasses", "yawn") **and your own videos** | `dataset/eval/...` |
Check each site's licence and cite it in your report. Verify the links yourself - they can change.

Your own data (best for the viva): record short videos, then
```
python training/extract_frames.py clip_open.mp4  dataset/eval/eye/open --every 15
python training/extract_frames.py clip_closed.mp4 dataset/eval/eye/closed --every 15
python training/extract_frames.py clip_yawn.mp4  dataset/eval/yawn/yawn --every 15
python training/extract_frames.py clip_noyawn.mp4 dataset/eval/yawn/no_yawn --every 15
python training/extract_frames.py clip_sun.mp4   dataset/eval/glasses/sunglasses --every 15
python training/extract_frames.py clip_clear.mp4 dataset/eval/glasses/clear --every 15
```
Use different people / lighting; aim for 300+ images per class. Full face visible.

## 2. Train the eye CNN (CPU is fine, minutes)
```
python training/train_drowsiness_cnn.py
```
Outputs: `models/eye_cnn.pth`, `reports/eye_cnn_history.png`, `eye_cnn_confusion_matrix.png`, `eye_cnn_metrics.json`.
Meaning: *epoch* = one pass over all training images; *loss* = how wrong the model is (lower is better);
*validation* = images used to choose the best epoch; *test* = images used once at the end for the honest score.
Healthy: train and validation curves both improve. If train accuracy is far above validation -> overfitting.

## 3. Validate / tune EAR, MAR and the sunglasses heuristic
```
python training/evaluate_real.py
```
Copy the printed thresholds into `CONFIG` in `app.py`. These are the numbers to quote.

## 4. Phone detector (optional)
The app works with stock YOLOv8n. To fine-tune: download the Roboflow dataset into `dataset/phone_data`,
set `"device": "cpu"` (or use Google Colab GPU) in `training/train_yolo_phone.py`, then
`python training/train_yolo_phone.py`. The best weights are copied to `yolov8_model/phone_detector_best.pt`
and the app uses them automatically. Report mAP50 / mAP50-95 for stock vs fine-tuned on the same val images.

## 5. Run the app
```
python app.py
```
Open http://127.0.0.1:5000 and register a new account (the demo database was removed on purpose).

## Still to do (not done in this package)
- The trained eye CNN is not yet wired into `app.py` (it still uses EAR). Integrate after training.
- A trained sunglasses classifier is not included; the heuristic is still used.
- Change `SECRET_KEY` in `app.py` before any real deployment.
