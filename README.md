# Tennis Analysis AI

This project aims to analyse tennis matches videos to extract valuable data, such as player position, ball speed, number of shots, etc.

## Models used

[YOLO11](https://docs.ultralytics.com/models/yolo11/) for ball detection

## Training

### Ball detection

#### Dataset

Training dataset from [Viren Dhanwani on Roboflow](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection)

You will need a Roboflow API key to download and use the dataset ([get your API key here](https://docs.roboflow.com/api-reference/authentication))

To use this key, create a `.env` file at the root of your project, and add `API_KEY=<YOUR_API_KEY_HERE>`

#### Training

Open `training/tennis_ball_detector_training.ipynb`

Run the Jupyter Notebook cells

You might need to adjust the `epochs`, `imgsz` and/or `batch` values depending on your available compute power and time requirements

(Model was run in a AWS SageMaker instance, with a `ml.g4dn.xlarge` notebook type and took ~1h to compute)

Download the outputed weights from `training/runs/detect/trainX/weights` and add them to the `models` folder

(You can find the trained weight I used [here](https://drive.google.com/file/d/1K3fRR7tliyxf82ckYCcajzftrB7SwWEg/view?usp=sharing))

#### Predicting

In `ball_prediction.py`, change the `model_ball` path to correspond to your computed model and run the script

Output is will be found in `runs/detect/predictX/`

### Object tracking

Run `object_tracking.py`

Output is will be found in `runs/detect/trackX/`

### Court key points

#### Dataset

Dataset found on [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) by [yastrebksv](https://github.com/yastrebksv)

#### Training

Open `training/tennis_court_keypoints_training.ipynb`

Run the Jupyter Notebook cells

(Model was run locally, compute time was ~5h for 20 epochs.\
Laptop specs: `Nvidia RTX 2060` GPU, `Intel Core i7-10750H` CPU and `32GB` of RAM)

Download the outputed model from `training/keypoints_model.pth` and add them to the `models` folder
