from copy import deepcopy

import cv2
import pandas as pd

import constants
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from trackers import BallTracker, PlayerTracker
from utils import (
    convert_pixel_distance_to_meters,
    measure_distance,
    read_video,
    save_video,
)


def main():
    # Split video into frames
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detection
    ## Detecting court line keypoints
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    ## Detecting players
    player_tracker = PlayerTracker(model_path="yolo11x")
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/player_detections.pkl",
    )

    ### Choose only players
    player_detections = player_tracker.choose_and_filter_players(
        court_keypoints, player_detections
    )

    ## Detecting ball
    ball_tracker = BallTracker(model_path="models/yolo11x_last.pt")
    ball_detections = ball_tracker.detect_frames(
        video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl"
    )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    ## Detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # Drawing Bounding Boxes
    ## Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    ## Draw Ball Bounding Boxes
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Draw Court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(
        output_video_frames, court_keypoints
    )

    # Mini court
    ## Initialize mini court
    mini_court = MiniCourt(video_frames[0])

    ## Convert player positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = (
        mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections, ball_detections, court_keypoints
        )
    )

    ## Draw mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)

    ## Draw player postitions on mini court
    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames, player_mini_court_detections
    )

    ## Draw ball postition on mini court
    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames, ball_mini_court_detections, color=(0, 255, 255)
    )

    # Player stats

    ## List to store statistical data related to player performance
    player_stats_data = [
        {
            "frame_num": 0,
            "player_1_number_of_shots": 0,
            "player_1_total_shot_speed": 0,
            "player_1_last_shot_speed": 0,
            "player_1_total_player_speed": 0,
            "player_1_last_player_speed": 0,
            "player_2_number_of_shots": 0,
            "player_2_total_shot_speed": 0,
            "player_2_last_shot_speed": 0,
            "player_2_total_player_speed": 0,
            "player_2_last_player_speed": 0,
        }
    ]

    ## Calculates various statistics related to the shot and the players on each ball hit
    for ball_shot_index in range(len(ball_shot_frames) - 1):
        # Get start and end frames for the current ball shot
        start_frame = ball_shot_frames[ball_shot_index]
        end_frame = ball_shot_frames[ball_shot_index + 1]

        # Calculate the time duration of the ball shot in seconds (assuming 24fps)
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24

        # Measure and convert the pixel distance covered by the ball to meters
        distance_covered_by_ball_pixels = measure_distance(
            ball_mini_court_detections[start_frame][1],
            ball_mini_court_detections[end_frame][1],
        )

        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
            distance_covered_by_ball_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court(),
        )

        # Calculate the speed of the ball shot in kilometers per hour (km/h)
        speed_of_ball_shot = (
            distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6
        )

        # Identify which player shot the ball based on their position at the start frame
        player_positions = player_mini_court_detections[start_frame]

        player_shot_ball = min(
            player_positions.keys(),
            key=lambda player_id: measure_distance(
                player_positions[player_id], ball_mini_court_detections[start_frame][1]
            ),
        )

        # Determine the opponent player ID based on who shot the ball
        opponent_player_id = 1 if player_shot_ball == 2 else 2

        # Measure and convert the pixel distance covered by the opponent to meters
        distance_covered_by_opponent_pixels = measure_distance(
            player_mini_court_detections[start_frame][opponent_player_id],
            player_mini_court_detections[end_frame][opponent_player_id],
        )

        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
            distance_covered_by_opponent_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court(),
        )

        # Calculate the speed of the opponent in kilometers per hour (km/h)
        speed_of_opponent = (
            distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6
        )

        # Create a copy of the current player statistics
        current_player_stats = deepcopy(player_stats_data[-1])

        # Update the frame number for these stats
        current_player_stats["frame_num"] = start_frame

        # Increment the count and update the total shot speed for the player who shot the ball
        current_player_stats[f"player_{player_shot_ball}_number_of_shots"] += 1

        current_player_stats[f"player_{player_shot_ball}_total_shot_speed"] += (
            speed_of_ball_shot
        )

        current_player_stats[f"player_{player_shot_ball}_last_shot_speed"] = (
            speed_of_ball_shot
        )

        # Update the total and last player speed for the opponent
        current_player_stats[f"player_{opponent_player_id}_total_player_speed"] += (
            speed_of_opponent
        )

        current_player_stats[f"player_{opponent_player_id}_last_player_speed"] = (
            speed_of_opponent
        )

        player_stats_data.append(current_player_stats)

    ## Convert the list of dictionaries containing player statistics data into a DataFrame
    player_stats_data_df = pd.DataFrame(player_stats_data)

    ## Create a DataFrame with frame numbers ranging from 0 to the length of video_frames minus one
    frames_df = pd.DataFrame({"frame_num": list(range(len(video_frames)))})

    ## Merge frames_df and player_stats_data_df on the "frame_num" column using an outer join (how="left")
    player_stats_data_df = pd.merge(
        frames_df, player_stats_data_df, on="frame_num", how="left"
    )

    ## Forward fill any missing values in the merged DataFrame
    player_stats_data_df = player_stats_data_df.ffill()

    ## Calculate average shot speed for Player 1
    player_stats_data_df["player_1_average_shot_speed"] = (
        player_stats_data_df["player_1_total_shot_speed"]
        / player_stats_data_df["player_1_number_of_shots"]
    )

    ## Calculate average shot speed for Player 2
    player_stats_data_df["player_2_average_shot_speed"] = (
        player_stats_data_df["player_2_total_shot_speed"]
        / player_stats_data_df["player_2_number_of_shots"]
    )

    ## Calculate average player speed for Player 1
    player_stats_data_df["player_1_average_player_speed"] = (
        player_stats_data_df["player_1_total_player_speed"]
        / player_stats_data_df["player_2_number_of_shots"]
    )

    ## Calculate average player speed for Player 2
    player_stats_data_df["player_2_average_player_speed"] = (
        player_stats_data_df["player_2_total_player_speed"]
        / player_stats_data_df["player_1_number_of_shots"]
    )

    ## Draw player stats
    def draw_player_stats(output_video_frames, player_stats):
        # Loop through each row in the player_stats DataFrame
        for index, row in player_stats.iterrows():
            # Extract shot speed and player speed data from current row
            player_1_shot_speed = row["player_1_last_shot_speed"]
            player_2_shot_speed = row["player_2_last_shot_speed"]
            player_1_speed = row["player_1_last_player_speed"]
            player_2_speed = row["player_2_last_player_speed"]

            # Extract average shot speed and player speed data from current row
            avg_player_1_shot_speed = row["player_1_average_shot_speed"]
            avg_player_2_shot_speed = row["player_2_average_shot_speed"]
            avg_player_1_speed = row["player_1_average_player_speed"]
            avg_player_2_speed = row["player_2_average_player_speed"]

            # Get the current frame from output_video_frames
            frame = output_video_frames[index]

            # Define dimensions of the overlay rectangle
            width = 350
            height = 230

            # Calculate the start and end coordinates for the overlay rectangle
            start_x = frame.shape[1] - 400
            start_y = frame.shape[0] - 500
            end_x = start_x + width
            end_y = start_y + height

            # Create a copy of the current frame to draw on
            overlay = frame.copy()

            # Draw a semi-transparent black rectangle over the specified area
            cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)

            # Combine the original and overlay frames using alpha blending
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            output_video_frames[index] = frame

            # Draw text labels for player statistics
            text = "     Player 1     Player 2"
            output_video_frames[index] = cv2.putText(
                output_video_frames[index],
                text,
                (start_x + 80, start_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Draw "Shot Speed" label
            text = "Shot Speed"
            output_video_frames[index] = cv2.putText(
                output_video_frames[index],
                text,
                (start_x + 10, start_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            # Draw actual shot speeds
            text = f"{player_1_shot_speed:.1f} km/h    {player_2_shot_speed:.1f} km/h"
            output_video_frames[index] = cv2.putText(
                output_video_frames[index],
                text,
                (start_x + 130, start_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            # Draw "Player Speed" label
            text = "Player Speed"
            output_video_frames[index] = cv2.putText(
                output_video_frames[index],
                text,
                (start_x + 10, start_y + 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            # Draw actual player speeds
            text = f"{player_1_speed:.1f} km/h    {player_2_speed:.1f} km/h"
            output_video_frames[index] = cv2.putText(
                output_video_frames[index],
                text,
                (start_x + 130, start_y + 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            # Draw "avg. S. Speed" label
            text = "avg. S. Speed"
            output_video_frames[index] = cv2.putText(
                output_video_frames[index],
                text,
                (start_x + 10, start_y + 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            # Draw average shot speeds
            text = f"{avg_player_1_shot_speed:.1f} km/h    {avg_player_2_shot_speed:.1f} km/h"
            output_video_frames[index] = cv2.putText(
                output_video_frames[index],
                text,
                (start_x + 130, start_y + 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            # Draw "avg. P. Speed" label
            text = "avg. P. Speed"
            output_video_frames[index] = cv2.putText(
                output_video_frames[index],
                text,
                (start_x + 10, start_y + 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            # Draw average player speeds
            text = f"{avg_player_1_speed:.1f} km/h    {avg_player_2_speed:.1f} km/h"
            output_video_frames[index] = cv2.putText(
                output_video_frames[index],
                text,
                (start_x + 130, start_y + 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        return output_video_frames

    ## Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    # Combines frames to video
    save_video(output_video_frames, "output_videos/output_video.mp4")


if __name__ == "__main__":
    main()
