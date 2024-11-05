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

In `yolo_inference.py` change the `model` path and run the script

Output is will be found in `runs/detect/predictX`
