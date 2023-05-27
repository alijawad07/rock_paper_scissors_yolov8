# Rock Paper Scissors with Object Detection

This project uses object detection to play rock paper scissors. A ***YOLOv8*** model is trained on a dataset of rock paper scissors images from ***Roboflow***. The model is then used to detect the player's hand gesture and determine the winner of the round.

Getting Started
To get started, clone the repository and install the dependencies:

git clone https://github.com/alijawad07/rock_paper_scissors_yolov8.git

## Prerequistes
You should already have ***Python 3.6+***
```bash
pip install -r requirements.txt
```

Code snippet

## Usage

To play the game, run the following command:
```
python3 rock_paper_scissors.py --data --source --output --weights
```
- --data => .yaml file with dataset and class details

- --source => Path to directory containing video

- --output => Path to save the detection results

- --weights => Path to yolov8 weights file


The model will then detect your gesture and determine the winner of the round.
