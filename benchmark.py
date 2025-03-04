import gc
import cv2
import numpy as np
from pathlib import Path
import openvino as ov
from ultralytics import YOLO
import urllib.request
import time

# A directory where the model will be downloaded.
model_name = "yolov8n"
det_model_path = Path(f"{model_name}_openvino_model/{model_name}.xml")

# export model to OpenVINO format using Ultralytics API
if not det_model_path.exists():
    pt_model = YOLO(f"{model_name}.pt")
    pt_model.export(format="openvino", dynamic=True, half=True)
    del pt_model
    gc.collect()

core = ov.Core()
device = "CPU"  # You can change this to "GPU" or other devices as needed

def load_model(det_model_path, device):
    compiled_model = compile_model(det_model_path, device)
    det_model = YOLO(det_model_path.parent, task="detect")

    if det_model.predictor is None:
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}
        args = {**det_model.overrides, **custom}
        det_model.predictor = det_model._smart_load("predictor")(overrides=args, _callbacks=det_model.callbacks)
        det_model.predictor.setup_model(model=det_model.model)

    det_model.predictor.model.ov_compiled_model = compiled_model
    return det_model

def compile_model(det_model_path, device):
    det_ov_model = core.read_model(det_model_path)

    ov_config = {}
    if device != "CPU":
        det_ov_model.reshape({0: [1, 3, 640, 640]})
    if "GPU" in device or ("AUTO" in device and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    det_compiled_model = core.compile_model(det_ov_model, device, ov_config)
    return det_compiled_model

det_model = load_model(det_model_path, device)

def run_object_detection(
    source=0,
    flip=False,
    use_popup=False,  # This parameter is kept for compatibility but no longer used
    skip_first_frames=0
):
    try:
        # Open video capture
        cap = cv2.VideoCapture(source)
        
        # Skip frames if needed
        for _ in range(skip_first_frames):
            cap.read()

        title = "Press ESC to Exit"
        cv2.namedWindow(winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Source ended")
                break

            if flip:
                frame = cv2.flip(frame, 1)

            # If the frame is larger than full HD, reduce size to improve performance
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(
                    src=frame,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )

            # Get the results
            input_image = np.array(frame)
            detections = det_model(input_image, verbose=False)
            frame = detections[0].plot()

            # Display the frame using OpenCV
            cv2.imshow(winname=title, mat=frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break

    except KeyboardInterrupt:
        print("Interrupted")
    except RuntimeError as e:
        print(e)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

# Example usage
USE_WEBCAM = False

video_file = Path("traffic.mp4")

source = 0 if USE_WEBCAM else str(video_file)
run_object_detection(source=source, flip=USE_WEBCAM)
