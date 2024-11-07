from ultralytics import YOLO

# Ball prediction

model_ball = YOLO('models/yolo11x_last.pt')
result = model_ball.predict('input_videos/input_video.mp4', conf=0.2, save=True)