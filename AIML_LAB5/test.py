# Install required packages before running:
# pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model (you can use 'yolov8n.pt', 'yolov8s.pt', etc.)
model = YOLO('yolov8n.pt')  # Use 'yolov8s.pt' for better accuracy

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run YOLOv8 detection and tracking
    results = model.track(frame, persist=True, show=False)

    # Annotate frame with bounding boxes, class labels, and tracking IDs
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Object Detection & Tracking", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()



import cv2
import numpy as np
from ultralytics import YOLO
import time

print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')
print("Model loaded successfully!")

# Function to perform object detection and tracking on a video file
def video_detection_tracking(video_path, save_output=False, output_path="output.avi"):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    
    # Optional: Save output video with detections
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, fps_input, (width, height))
    else:
        out = None
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("End of video file reached.")
                break
            
            frame_count += 1
            
            # Perform detection & tracking
            results = model.track(frame, persist=True, verbose=False)
            
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                track_ids = results[0].boxes.id
                
                if track_ids is not None:
                    track_ids = track_ids.cpu().numpy()
                
                for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                    if conf > 0.5:
                        x1, y1, x2, y2 = map(int, box)
                        class_name = model.names[int(cls)]
                        track_id = int(track_ids[i]) if track_ids is not None else None
                        
                        if track_id is not None:
                            label = f"{class_name} ID:{track_id} {conf:.2f}"
                            color = (0, 255, 0)
                        else:
                            label = f"{class_name} {conf:.2f}"
                            color = (0, 0, 255)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                      (x1 + label_size[0], y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Calculate FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("YOLOv8 Video Object Detection & Tracking", frame)
            
            if out is not None:
                out.write(frame)
            
            # Press 'q' to stop early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        print("Video processing complete.")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {fps:.2f}")


# Run detection on a video file
video_path = "video.mp4"  # Replace with your video file path
video_detection_tracking(video_path, save_output=True, output_path="detected_output.avi")

