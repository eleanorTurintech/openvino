#!/usr/bin/env python3
"""
Pure OpenVINO Benchmark Script
- Completely avoids cv2/OpenCV dependencies
- Works with locally built OpenVINO
- Doesn't modify any installed packages
"""

import os
import sys
import gc
import time
import statistics
import numpy as np
from pathlib import Path
import urllib.request

# First let's check if av is available for video processing
try:
    import av
    AV_AVAILABLE = True
    print("PyAV is available for video processing")
except ImportError:
    AV_AVAILABLE = False
    print("WARNING: PyAV not available. Install with 'pip install av' for video processing")
    print("Attempting to use PIL for image processing instead")
    try:
        from PIL import Image
        PIL_AVAILABLE = True
    except ImportError:
        PIL_AVAILABLE = False
        print("WARNING: Neither PyAV nor PIL is available")

# Import the local OpenVINO
import openvino as ov
print(f"Using OpenVINO version: {ov.__version__}")

# Check if ultralytics is available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: Ultralytics not available. Install with 'pip install ultralytics' for YOLO models")

# Pure Python video capture class using PyAV
class AVVideoCapture:
    """Video capture implementation using PyAV instead of OpenCV"""
    def __init__(self, path):
        if not AV_AVAILABLE:
            raise ImportError("PyAV is required for video processing. Install with 'pip install av'")
        
        self.path = str(path)
        try:
            self.container = av.open(self.path)
            self.stream = next(s for s in self.container.streams if s.type == 'video')
            self.fps = float(self.stream.average_rate)
            self.width = self.stream.width
            self.height = self.stream.height
            self.frame_count = self.stream.frames if self.stream.frames > 0 else 1000  # Default if unknown
            self.frames_iter = self.container.decode(video=0)
            print(f"Successfully opened video with PyAV: {self.width}x{self.height} @ {self.fps} fps, {self.frame_count} frames")
        except Exception as e:
            print(f"Error opening video with PyAV: {e}")
            raise
        
    def isOpened(self):
        return True
            
    def get(self, prop_id):
        # Map OpenCV property IDs to our values
        if prop_id == 5:  # CAP_PROP_FPS
            return self.fps
        elif prop_id == 7:  # CAP_PROP_FRAME_COUNT
            return self.frame_count
        elif prop_id == 3:  # CAP_PROP_FRAME_WIDTH
            return self.width
        elif prop_id == 4:  # CAP_PROP_FRAME_HEIGHT
            return self.height
        return 0
            
    def read(self):
        try:
            frame = next(self.frames_iter)
            # Convert PyAV frame to numpy array (BGR format)
            img = frame.to_ndarray(format='rgb24')
            # Convert RGB to BGR (OpenCV format) for compatibility with YOLO
            img = img[:, :, ::-1].copy()
            return True, img
        except StopIteration:
            return False, None
        except Exception as e:
            print(f"Error reading frame: {e}")
            return False, None
                
    def release(self):
        try:
            self.container.close()
        except:
            pass

# Pure Python image resizing function
def resize_image(image, target_size=None, scale=None):
    """Resize image using numpy operations"""
    h, w = image.shape[:2]
    
    if target_size is not None:
        target_w, target_h = target_size
    elif scale is not None:
        target_h, target_w = int(h * scale), int(w * scale)
    else:
        return image
    
    # Create output array
    resized = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    
    # Calculate scaling factors
    x_ratio = float(w - 1) / (target_w - 1) if target_w > 1 else 0
    y_ratio = float(h - 1) / (target_h - 1) if target_h > 1 else 0
    
    # Use numpy operations for faster processing
    y_indices = np.floor(np.arange(target_h) * y_ratio).astype(int)
    x_indices = np.floor(np.arange(target_w) * x_ratio).astype(int)
    
    # Simple nearest-neighbor resizing
    for i in range(3):  # For each channel
        resized[:, :, i] = image[y_indices[:, np.newaxis], x_indices, i]
    
    return resized

def load_model(model_name, device, no_modify=True):
    """Load a YOLO model for OpenVINO inference without modifying installed packages"""
    print(f"Loading {model_name} on {device}...")
    
    if not YOLO_AVAILABLE:
        print("ERROR: Ultralytics YOLO is required for model loading")
        sys.exit(1)
    
    det_model_path = Path(f"{model_name}_openvino_model/{model_name}.xml")
    
    # Export model to OpenVINO format if needed
    if not det_model_path.exists():
        print(f"Exporting {model_name} to OpenVINO format...")
        pt_model = YOLO(f"{model_name}.pt")
        
        if no_modify:
            # Create a temporary directory for export to avoid modifying the installed package
            export_dir = Path(f"./tmp_export_{model_name}")
            export_dir.mkdir(exist_ok=True)
            pt_model.export(format="openvino", dynamic=True, half=True, export_dir=export_dir)
            
            # Move exported files to the desired location
            if export_dir.exists():
                target_dir = Path(f"{model_name}_openvino_model")
                target_dir.mkdir(exist_ok=True)
                for file in export_dir.glob("*"):
                    target_file = target_dir / file.name
                    if target_file.exists():
                        target_file.unlink()
                    file.rename(target_file)
                
                # Clean up
                import shutil
                try:
                    shutil.rmtree(export_dir, ignore_errors=True)
                except:
                    pass
        else:
            # Direct export
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
    
    # Use a custom predictor setup to avoid modifying the installed package
    custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}
    args = {**det_model.overrides, **custom}
    
    # Only setup the predictor if it's not already set up
    if det_model.predictor is None:
        try:
            det_model.predictor = det_model._smart_load("predictor")(overrides=args, _callbacks=det_model.callbacks)
            det_model.predictor.setup_model(model=det_model.model)
        except Exception as e:
            print(f"Error setting up predictor: {e}")
            raise
    
    # Set the compiled model without modifying the OpenVINO module
    det_model.predictor.model.ov_compiled_model = det_compiled_model
    return det_model

def run_benchmark(model_name, device, video_path, warmup_frames=10, max_frames=0, no_modify=True):
    """Run benchmark with given model, device and video"""
    # Load the model
    det_model = load_model(model_name, device, no_modify)
    
    # Open video source
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video file '{video_path}' not found")
        return
    
    # Open video using PyAV
    print(f"Opening video: {video_path}")
    try:
        cap = AVVideoCapture(str(video_path))
    except Exception as e:
        print(f"Error opening video: {e}")
        return
    
    # Get video details
    video_fps = cap.get(5)  # FPS
    total_frames = int(cap.get(7))  # Frame count
    width = int(cap.get(3))  # Width
    height = int(cap.get(4))  # Height
    
    print(f"Video: {video_path.name}, Resolution: {width}x{height}, FPS: {video_fps:.2f}, Frames: {total_frames}")
    print(f"Model: {model_name}, Device: {device}")
    print("Running in fully headless mode with PyAV for video processing")
    
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
        try:
            _ = det_model(frame, verbose=False)
        except Exception as e:
            print(f"Error during warmup: {e}")
            return
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
            # Use our custom resize function
            frame = resize_image(frame, scale=scale)
            
        # Perform inference and time it separately
        inference_start = time.time()
        try:
            detections = det_model(frame, verbose=False)
        except Exception as e:
            print(f"Error during inference at frame {frame_count}: {e}")
            break
        inference_end = time.time()
        
        # Store timing data
        inference_time = inference_end - inference_start
        inference_times.append(inference_time)
        
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
    
    # Process and report results
    if frame_count == 0:
        print("No frames processed. Exiting.")
        return
    
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

def download_sample_video(path="traffic.mp4"):
    """Download sample video if it doesn't exist"""
    video_path = Path(path)
    if not video_path.exists():
        print(f"Downloading sample video to {path}...")
        video_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/traffic.mp4"
        try:
            urllib.request.urlretrieve(video_url, video_path)
            print("Download complete!")
        except Exception as e:
            print(f"Error downloading video: {e}")
            return False
    return True

if __name__ == "__main__":
    # Benchmark configuration
    MODEL_NAME = "yolov8n"           # Model to benchmark (yolov8n, yolov8s, yolov8m, etc.)
    DEVICE = "CPU"                   # Device to use (CPU, GPU, AUTO, etc.)
    VIDEO_PATH = "traffic.mp4"       # Path to video file
    WARMUP_FRAMES = 10               # Number of frames to use for warmup
    MAX_FRAMES = 100                 # Maximum frames to process (0 for all)
    NO_MODIFY = True                 # Set to True to avoid modifying installed packages
    
    # Download the video file if it doesn't exist
    if VIDEO_PATH == "traffic.mp4":
        if not download_sample_video(VIDEO_PATH):
            sys.exit(1)
    
    # Run the benchmark in headless mode
    print("Starting benchmark in fully headless mode...")
    try:
        run_benchmark(
            model_name=MODEL_NAME,
            device=DEVICE,
            video_path=VIDEO_PATH,
            warmup_frames=WARMUP_FRAMES,
            max_frames=MAX_FRAMES,
            no_modify=NO_MODIFY
        )
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Error during benchmark: {e}")