from utils import (read_video, save_video)
from trackers import PlayerTracker

def main():
    # Split video into frames
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detecting players
    player_tracker = PlayerTracker(model_path='yolo11x')
    player_detections = player_tracker.detect_frames(video_frames)

    # Combines frames to video
    save_video(video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()