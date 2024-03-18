# Importing necessary libraries
from ultralytics import YOLO  # Importing the YOLO object detection model from Ultralytics
import cv2  # Importing OpenCV for image processing tasks
import numpy as np  # Importing NumPy for numerical computations
import csv  # Importing CSV module for reading and writing CSV files

# Initializing YOLOv8 object detection model
object_detector = YOLO('yolov8n.pt')  # Initializing YOLOv8 model with pre-trained weights ('yolov8n.pt')

# Path to the input video file
input_video_path = "Videos/sample_video2.MOV"  # Path to the input video file
video_capture = cv2.VideoCapture(input_video_path)  # Opening the video file for reading

# Get frames per second (fps) of the input video
fps = video_capture.get(cv2.CAP_PROP_FPS)  # Retrieve frames per second (fps) of the input video

# Output video file path
output_video_path = "Videos/Output/output.mp4"  # Path to the output video file

# Coordinates of Region of Interest (ROI)
region_of_interest_Coordinates = np.array([[22, 154], [1602, 170], [1630, 1066], [42, 1054]], dtype=np.int32)
# Define the coordinates of the Region of Interest (ROI) rectangles where people will be detected and tracked

# Dictionary to store entry timestamps of people in the queue
people_entry_timestamps = {}  # Dictionary to store entry timestamps of people in the queue

# List to store time spent by each person in the queue
time_spent_in_queue = []  # List to store time spent by each person in the queue

# CSV file to store time spent by each person in the queue
csv_filename = "queue_time.csv"  # Name of the CSV file to store time spent by each person in the queue

# Open the CSV file in write mode
csv_file = open(csv_filename, 'w', newline='')  # Open the CSV file for writing

# Create CSV writer object
csv_writer = csv.writer(csv_file)  # Create a CSV writer object to write data to the CSV file

# Initialize frame count
frame_count = 0  # Initialize frame count to keep track of the number of frames processed

# Get dimensions of the input video
video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the input video frame
video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the input video frame

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for the output video
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (video_width, video_height)) # Create a VideoWriter object to write the annotated frames to the output video

# Loop through each frame in the video
while video_capture.isOpened():  # Iterate until the video ends or there are no more frames

    # Read a frame from the video
    success, frame = video_capture.read()  # Read a frame from the input video

    if success:  # Check if the frame was read successfully

        # Perform object detection and tracking using YOLOv8
        detection_results = object_detector.track(frame, persist=True) # Detect and track objects (people) in the frame using YOLOv8 model with persistence enabled

        # Extract bounding boxes and track IDs
        detected_boxes = detection_results[0].boxes.xyxy.cpu()  # Extract detected bounding boxes from detection results
        track_ids = detection_results[0].boxes.id.int().cpu().tolist() # Extract track IDs associated with the detected bounding boxes

        # Check if the center of bounding boxes is inside the region of interest (ROI)
        for box, track_id in zip(detected_boxes, track_ids):
            x1, y1, x2, y2 = box  # Extract bounding box coordinates (top-left and bottom-right corners)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert coordinates to integers

            x_center = (x1 + x2) / 2  # Calculate x-coordinate of the center of the bounding box
            y_center = (y1 + y2) / 2  # Calculate y-coordinate of the center of the bounding box

            # Check if the center of the bounding box is inside the ROI
            if cv2.pointPolygonTest(region_of_interest_Coordinates, (x_center, y_center), False) > 0:
                if str(track_id) not in people_entry_timestamps:
                    # Record entry timestamp of the person in the queue
                    people_entry_timestamps[str(track_id)] = str(frame_count)

                # Draw rectangle around the detected person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person ID: " + str(track_id), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (20, 255, 0), 2)
                # Annotate the frame with bounding box and person ID

            else:  # If the person exits the ROI
                if str(track_id) in people_entry_timestamps:
                    # Record exit timestamp of the person from the queue
                    exit_timestamp = frame_count
                    entry_timestamp = people_entry_timestamps[str(track_id)]

                    # Calculate time spent by the person in the queue
                    time_spent = (exit_timestamp - int(entry_timestamp)) / fps
                    time_spent_in_queue.append(time_spent)

                    # Write time spent by the person to the CSV file
                    csv_writer.writerow(
                        ["Time spent by person " + str(track_id) + " in line is " + str(time_spent)])

                    # Remove entry from the dictionary
                    people_entry_timestamps.pop(str(track_id))

        # Draw ROI rectangles on the frame
        cv2.drawContours(frame, [region_of_interest_Coordinates], -1, (255, 0, 0), 3)
        # Draw the region of interest (ROI) rectangles on the frame for visualization

        # Write annotated frame to the output video
        output_video.write(frame)  # Write the annotated frame to the output video

        # Increment frame count
        frame_count += 1  # Increment the frame count for the next frame

    else:  # If there are no more frames in the video
        break  # Break the loop

# Release video capture, output video, and CSV file resources
video_capture.release()  # Release the video capture object
output_video.release()  # Release the output video object
csv_file.close()  # Close the CSV file

# Print the path to the output video file
print(f"Output video saved at: {output_video_path}")  # Print the path to the output video file
