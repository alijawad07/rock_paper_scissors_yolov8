import cv2
import time
from ultralytics import YOLO
import argparse
import yaml
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on test images')
    parser.add_argument('--data', required=True, help='Path to YAML config file')
    parser.add_argument('--source', required=True, help='Path to directory containing images')
    parser.add_argument('--output', required=True, help='Path to save the inference result')
    parser.add_argument('--weights',required=True, help='Path to checkpoint file')
    return parser.parse_args()
    
args = parse_args()

# Load the model
model = YOLO(args.weights)

#create output directory if it doesnot exist
os.makedirs(args.output, exist_ok=True)


# Load dataset parameters from YAML config file
with open(args.data, 'r') as f:
    config = yaml.safe_load(f)
classes = config['names']


# Initialize the win and loss counts
left_wins = 0
right_wins = 0
left_losses = 0
right_losses = 0

# Start the video capture
cap = cv2.VideoCapture(args.source)

# Get the current time
start_time = time.time()
frame_count = 0

# Get the video frame rate and dimensions
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_video = cv2.VideoWriter(os.path.join(args.output, "output.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

last_detection = (255, 255, 255),(255, 255, 255)

while True:
    # Capture the frame
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection
    results = model(frame)[0]

    # Get the detections as a list of dictionaries
    detections = results.boxes.data.tolist()

    # Find the left and right players
    left_player = None
    right_player = None

    for detection in detections:
        xmin, ymin, xmax, ymax, confidence, class_idx = detection

        if confidence > 0.9:  # Apply confidence threshold
            if xmin < (frame.shape[1] // 2):
                left_player = detection
            else:
                right_player = detection

    left_color = (128, 128, 128)  # Gray
    right_color = (128, 128, 128)  # Gray
    
    # If both players are found, referee the game
    if left_player and right_player:
        
        # Get the objects that were detected
        left_object = classes[int(left_player[5])]
        right_object = classes[int(right_player[5])]
 
        # Determine the winner
        if (left_object == "Rock" and right_object == "Scissors") or (left_object == "Paper" and right_object == "Rock") or (left_object == "Scissors" and right_object == "Paper"):
            left_color = (0, 255, 0)  # Green
            right_color = (0, 0, 255)  # Red
            if (right_color == (0, 0, 255) and left_color == (0, 255, 0)) and (last_detection == ((128, 128, 128),(128, 128, 128))):
                left_wins += 1
                right_losses += 1
                last_detection = (left_color, right_color)
            
            
        elif (right_object == "Rock" and left_object == "Scissors") or (right_object == "Paper" and left_object == "Rock") or (right_object == "Scissors" and left_object == "Paper"):
            left_color = (0, 0, 255)  # Red
            right_color = (0, 255, 0)  # Green
            if (left_color == (0, 0, 255) and right_color == (0, 255, 0)) and (last_detection == ((128, 128, 128),(128, 128, 128))):
                right_wins += 1
                left_losses += 1
                last_detection = (left_color, right_color)
            
            
        else:
            left_color = (128, 128, 128)  # Gray
            right_color = (128, 128, 128)  # Gray
            last_detection = (left_color, right_color)


        # Draw bounding boxes for detections
        xmin, ymin, xmax, ymax, _, class_idx = left_player
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), left_color, 2)
        cv2.putText(frame, classes[int(class_idx)], (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 2)
        xmin, ymin, xmax, ymax, _, class_idx = right_player
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), right_color, 2)
        cv2.putText(frame, classes[int(class_idx)], (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 2)

        frame_count += 1
        
        left_color = None
        right_color = None

    # Display the results
    cv2.putText(frame, "Left Score: {} | Right Score: {}".format(left_wins, right_wins), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    #cv2.putText(frame, "Left Lost: {} | Right Lost: {}".format(left_losses, right_losses), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    cv2.putText(frame, "Model: Yolov8x", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 138, 139),3)

    # Get the current time
    end_time = time.time()

    # Calculate the FPS
    fps = int(frame_count / (end_time - start_time))

    # Display the FPS
    cv2.putText(frame, "FPS: {}".format(fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

    # Write the frame to the output video
    out_video.write(frame)



cap.release()
out_video.release()
