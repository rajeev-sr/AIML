import cv2
import numpy as np
from ultralytics import YOLO
from transformers import pipeline
import whisper
import ffmpeg
import os

# -----------------------------
# Load models
# -----------------------------
print("Loading models...")
yolo_model = YOLO("yolov8n.pt")  # YOLOv8 for object detection
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")  # Text classification
whisper_model = whisper.load_model("base")  # Speech-to-text
print("Models loaded successfully!")

# -----------------------------
# Step 1: Extract audio from video
# -----------------------------
def extract_audio(video_path, audio_path="temp_audio.wav"):
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, format="wav", acodec="pcm_s16le", ac=1, ar="16k")
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"Audio extracted: {audio_path}")
        return audio_path
    except Exception as e:
        print("Error extracting audio:", e)
        return None

# -----------------------------
# Step 2: Transcribe audio with Whisper
# -----------------------------
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    text = result["text"]
    print("Transcription:", text[:200], "..." if len(text) > 200 else "")
    return text

# -----------------------------
# Step 3: Classify text with Zero-Shot
# -----------------------------
def classify_text(text, candidate_labels=None):
    if candidate_labels is None:
        candidate_labels = ["sports", "politics", "technology", "education", "entertainment", "business"]
    classification = classifier(text, candidate_labels=candidate_labels)
    print("Classification result:", classification["labels"][0], "→ score:", classification["scores"][0])
    return classification

# -----------------------------
# Step 4: Detect objects in video with YOLOv8
# -----------------------------
def detect_objects(video_path, output_path="detected_output.avi"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Running YOLOv8 object detection on video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = yolo_model(frame)
        annotated_frame = results[0].plot()  # YOLO built-in annotation
        
        out.write(annotated_frame)
        cv2.imshow("YOLOv8 Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Detection finished. Saved to {output_path}")

# -----------------------------
# Run full pipeline
# -----------------------------
video_path = "video.mp4"  # 🔹 replace with your video file
audio_path = extract_audio(video_path)

if audio_path:
    text = transcribe_audio(audio_path)
    classify_text(text)
    detect_objects(video_path, "video_with_objects.avi")

