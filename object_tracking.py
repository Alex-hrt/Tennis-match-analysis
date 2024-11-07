from ultralytics import YOLO

# Object tracking

model_track = YOLO('yolo11x')
result = model_track.track('input_videos/input_video.mp4', conf=0.2, save=True)