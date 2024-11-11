from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector


def main():
    # Split video into frames
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detection
    ## Detecting players
    player_tracker = PlayerTracker(model_path="yolo11x")
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/player_detections.pkl",
    )

    ## Detecting ball
    ball_tracker = BallTracker(model_path="models/yolo11x_last.pt")
    ball_detections = ball_tracker.detect_frames(
        video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl"
    )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    ## Detecting Court line keypoints
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Drawing Bounding Boxes
    ## Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    ## Draw Ball Bounding Boxes
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Draw Court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(
        output_video_frames, court_keypoints
    )

    # Combines frames to video
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
