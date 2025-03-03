import cv2
import numpy as np
import time
from openvino.runtime import Core

# Load YOLOv8 model with OpenVINO
core = Core()
model = core.read_model("yolov8n.xml")
compiled_model = core.compile_model(model, "CPU")

# Set up video capture (webcam or video file)
cap = cv2.VideoCapture("sorting.mp4")  # or 0 for webcam

# Performance metrics tracking
baseline_fps = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    start_time = time.time()
    
    # Pre-process frame
    input_image = preprocess(frame)
    
    # Run inference
    results = compiled_model(input_image)[0]
    
    # Post-process results
    processed_frame = postprocess(frame, results)
    
    # Calculate FPS
    fps = 1 / (time.time() - start_time)
    baseline_fps.append(fps)
    
    # Display results
    cv2.putText(processed_frame, f"FPS: {fps:.2f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Baseline OpenVINO", processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Average baseline FPS: {np.mean(baseline_fps):.2f}")