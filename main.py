from utils import (read_video, save_video)
from trackers import PlayerTracker, BallTracker

def main():
    # Split video into frames
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detection
    ## Detecting players
    player_tracker = PlayerTracker(model_path='yolo11x')
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")

    ## Detecting balls
    ball_tracker = BallTracker(model_path='models/yolo11x_last.pt')
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")
    
    # Drawing Bounding Boxes
    ## Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    ## Draw Ball Bounding Boxes
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)

    # Combines frames to video
    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()