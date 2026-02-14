from dataclasses import dataclass


@dataclass(frozen=True)
class VisionConfig:
    # Camera
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720

    # Processing
    target_fps: int = 30
    yolo_every_n: int = 10

    # Gaze — iris horizontal ratio
    iris_ratio_min: float = 0.28
    iris_ratio_max: float = 0.72

    # Gaze — head pitch degrees, negative = looking down
    head_pitch_threshold: float = -15.0

    # YOLO
    yolo_model: str = "yolo11n.pt"
    yolo_device: str = "mps"
    yolo_confidence: float = 0.45
    yolo_phone_class: int = 67  # COCO "cell phone"

    # MediaPipe
    face_model_path: str = "models/face_landmarker.task"

    # State machine — seconds
    confirm_duration: float = 2.0
    clear_duration: float = 1.0
