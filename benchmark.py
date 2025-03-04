import gc
import cv2
import numpy as np
from pathlib import Path
import openvino as ov
from ultralytics import YOLO
import urllib.request
import time
import os
import statistics

# Function to check if display is available
def is_display_available():
    try:
        test_window = "Test"
        cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(test_window)
        return True
    except:
        return False

def load_model(model_name, device):
    print(f"Loading {model_name} on {device}...")
    det_model_path = Path(f"{model_name}_openvino_model/{model_name}.xml")
    
    # Export model to OpenVINO format if needed
    if not det_model_path.exists():
        print(f"Exporting {model_name} to OpenVINO format...")
        pt_model = YOLO(f"{model_name}.pt")
        pt_model.export(format="openvino", dynamic=True, half=True)
        del pt_model
        gc.collect()
    
    # Load and compile the model
    core = ov.Core()
    det_ov_model = core.read_model(det_model_path)
    
    ov_config = {}
    if device != "CPU":
        det_ov_model.reshape({0: [1, 3, 640, 640]})
    if "GPU" in device or ("AUTO" in device and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    
    det_compiled_model = core.compile_model(det_ov_model, device, ov_config)
    
    # Create YOLO model with OpenVINO backend
    det_model = YOLO(det_model_path.parent, task="detect")
    if det_model.predictor is None:
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}
        args = {**det_model.overrides, **custom}
        det_model.predictor = det_model._smart_load("predictor")(overrides=args, _callbacks=det_model.callbacks)
        det_model.predictor.setup_model(model=det_model.model)
    
    det_model.predictor.model.ov_compiled_model = det_compiled_model
    return det_model

def run_benchmark(model_name, device, video_path, warmup_frames=10, disable_display=False, max_frames=0):
    # Load the model
    det_model = load_model(model_name, device)
    
    # Check if display is available and enabled
    has_display = is_display_available() and not disable_display
    if not has_display:
        print("Running in headless mode")
    
    # Open video source
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video file '{video_path}' not found")
        return
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        return
    
    # Get video details
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_path.name}, Resolution: {width}x{height}, FPS: {video_fps:.2f}, Frames: {total_frames}")
    print(f"Model: {model_name}, Device: {device}")
    
    # Create display window if available
    if has_display:
        window_title = f"Benchmark: {model_name} on {device}"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    
    # Prepare for timing
    frame_times = []
    inference_times = []
    frame_count = 0
    
    # Run warmup phase
    print(f"Starting warmup phase ({warmup_frames} frames)...")
    while frame_count < warmup_frames:
        ret, frame = cap.read()
        if not ret:
            print("Video ended during warmup")
            return
        
        # Process frame for warmup
        _ = det_model(frame, verbose=False)
        frame_count += 1
    
    print("Warmup complete. Starting benchmark...")
    
    # Reset counters and start timing
    frame_count = 0
    benchmark_start_time = time.time()
    
    # Process frames for benchmark timing
    frames_to_process = max_frames if max_frames > 0 else float('inf')
    
    while frame_count < frames_to_process:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Video ended")
            break
        
        # Record frame start time
        frame_start_time = time.time()
        
        # Preprocess frame
        scale = 1280 / max(frame.shape)
        if scale < 1:
            frame = cv2.resize(
                src=frame,
                dsize=None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
            
        # Perform inference and time it separately
        inference_start = time.time()
        detections = det_model(frame, verbose=False)
        inference_end = time.time()
        
        # Store timing data
        inference_time = inference_end - inference_start
        inference_times.append(inference_time)
        
        # Process results
        if has_display:
            result_frame = detections[0].plot()
            cv2.imshow(window_title, result_frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                print("Benchmark interrupted")
                break
        
        # Record frame end time and add to statistics
        frame_end_time = time.time()
        frame_time = frame_end_time - frame_start_time
        frame_times.append(frame_time)
        
        # Show progress periodically
        frame_count += 1
        if frame_count % 10 == 0:
            current_fps = 1.0 / (sum(frame_times[-10:]) / 10)
            print(f"Frame {frame_count}: Current FPS: {current_fps:.2f}")
    
    # Calculate benchmark results
    benchmark_duration = time.time() - benchmark_start_time
    
    # Clean up
    cap.release()
    if has_display:
        cv2.destroyAllWindows()
    
    # Process and report results
    avg_fps = frame_count / benchmark_duration
    avg_inference_time = sum(inference_times) / len(inference_times)
    avg_frame_time = sum(frame_times) / len(frame_times)
    
    # Additional statistics
    max_frame_time = max(frame_times)
    min_frame_time = min(frame_times)
    median_frame_time = statistics.median(frame_times)
    
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Video: {video_path.name} ({width}x{height})")
    print(f"Frames processed: {frame_count}")
    print(f"Total benchmark time: {benchmark_duration:.2f} seconds")
    print("-"*50)
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average inference time: {avg_inference_time*1000:.2f} ms")
    print(f"Average frame time: {avg_frame_time*1000:.2f} ms")
    print(f"Min/Max/Median frame time: {min_frame_time*1000:.2f}/{max_frame_time*1000:.2f}/{median_frame_time*1000:.2f} ms")
    print("="*50)
    
    # Return statistics for potential additional processing
    return {
        "avg_fps": avg_fps,
        "avg_inference_time": avg_inference_time,
        "avg_frame_time": avg_frame_time,
        "frames_processed": frame_count,
        "benchmark_duration": benchmark_duration,
        "min_frame_time": min_frame_time,
        "max_frame_time": max_frame_time,
        "median_frame_time": median_frame_time,
    }

# Benchmark configuration - adjust these values as needed
MODEL_NAME = "yolov8n"           # Model to benchmark (yolov8n, yolov8s, yolov8m, etc.)
DEVICE = "CPU"                   # Device to use (CPU, GPU, AUTO, etc.)
VIDEO_PATH = "traffic.mp4"       # Path to video file
WARMUP_FRAMES = 10               # Number of frames to use for warmup
DISABLE_DISPLAY = False          # Set to True to run without display
MAX_FRAMES = 0                   # Maximum frames to process (0 for all)

# Download the video file if it doesn't exist and using the default traffic.mp4
video_path = Path(VIDEO_PATH)
if not video_path.exists() and video_path.name == "traffic.mp4":
    print("Downloading sample traffic video...")
    video_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/traffic.mp4"
    urllib.request.urlretrieve(video_url, video_path)

# Run the benchmark
run_benchmark(
    model_name=MODEL_NAME,
    device=DEVICE,
    video_path=VIDEO_PATH,
    warmup_frames=WARMUP_FRAMES,
    disable_display=DISABLE_DISPLAY,
    max_frames=MAX_FRAMES
)
