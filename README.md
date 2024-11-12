
# Tennis Match Analysis

## Overview

The **Tennis Match Analysis** project is designed to automate the analysis of tennis matches.\
It combines machine learning, computer vision, and data processing to provide insights into player performance, ball trajectory, and overall match dynamics.
## Project Structure

```
Tennis_match_analysis
├── constants/
│   └── __init__.py
├── court_line_detector/
│   ├── __init__.py
│   └── court_line_detector.py
├── input_videos/
│   ├── image.png
│   └── input_video.mp4
├── mini_court/
│   ├── __init__.py
│   └── mini_court.py
├── models/
│   └── ...  # Pre-trained and custom trained models can be stored here
├── output_videos/
│   └── ...  # Output videos will be stored here when running main.py
├── tracker_stubs/
│   ├── ball_detections.pkl
│   └── player_detections.pkl
├── trackers/
│   ├── __init__.py
│   ├── ball_tracker.py
│   └── player_tracker.py
├── training/
│   ├── tennis_ball_detector_training.ipynb
│   └── tennis_court_keypoints_training.ipynb
├── utils/
│   ├── __init__.py
│   ├── bbox_utils.py
│   ├── conversions.py
│   └── video_utils.py
├── README.md
├── ball_prediction.py
├── main.py
└── object_tracking.py
```
## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Alex-hrt/Tennis-match-analysis.git
   cd Tennis_match_analysis
   ```
2. **Set up a virtual environment:**
    - Install `virtualenv` if needed
        ```bash
        pip install virtualenv
        ```
    - Create a virtual environment named .venv
        ```bash
        virtualenv .venv
        ```
    - Activate the environment (use the correct command for your OS)
        #### macOS/Linux
        ```bash
        source .venv/bin/activate
        ```
        #### Windows
        ```bash
        .venv\Scripts\activate
        ```
    - To deactivate the environment
        ```bash
        deactivate
        ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download My Pre-trained Models:**\
   The pre-trained models are stored in the `models` directory.\
   You can find the weights and models I trained and used [here ↗](https://drive.google.com/drive/folders/1_TUux9y6OBNFzoBOQ9DRf1RV0XEHV_LI?usp=sharing)
    
## Running the Project

1. **Prepare Input Videos:**\
   Place your input videos in the `input_videos` directory. Ensure they are in a format that OpenCV can read (e.g., MP4, AVI).\
   (A default video is already provided)

2. **Make sure things are working:**
    - **Object Tracking:**\
        The `object_tracking.py` script uses a YOLO model to track objects in the video frames. Ensure the model is correctly loaded and configured by running it.\
        Prediction output can be found in `runs/detect/trackX`
    
    - **Ball Prediction:**\
        The `ball_prediction.py` script predicts ball positions using another pre-trained YOLO model. Adjust the confidence threshold as needed.\
        Prediction output can be found in `runs/detect/predictX`

3. **Run Main Script:**\
   Execute the main script to perform the analysis:
   ```bash
   python main.py
   ```
   The `main.py` script combines all detected elements into a final video with statistics and court analysis overlays. Prediction output can be found in `output_videos/`\
   (Pre-determined detections are stored in the `tracker_stubs` directory for future use)

## Training Custom Models

If you need to train custom models, refer to the Jupyter notebooks in the `training` directory. These notebooks guide you through the process of training models for ball detection and court keypoints.

Once training is completed you can run `object_tracking.py`, `ball_prediction.py` and/or `main.py` to see the result of your training\
(In `ball_prediction.py`, change the `model_ball` path to correspond to your computed model and run the script)

⚠️ **Ball detection training** ran in an AWS SageMaker instance, with a `ml.g4dn.xlarge` notebook type and took ~1h to compute for 100 epochs

⚠️ **Object tracking training:** ran locally (`Nvidia RTX 2060` GPU, `Intel Core i7-10750H` CPU, `32GB` of RAM), compute time was ~5h for 20 epochs