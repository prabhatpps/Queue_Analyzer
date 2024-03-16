# import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
import csv
import imageio

# Configure the tracking parameters and run the tracker
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "Videos/IMG_3804.MOV"
cap = cv2.VideoCapture(video_path)

fps = 29.98

# Define the output video path
output_video_path = "Videos/Output/output.mp4"

# Define ROI (Region of Interest) coordinates for the rectangular regions
roi1_coords = np.array([[22, 154], [1602, 170], [1630, 1066], [42, 1054]], dtype=np.int32)
roi2_coords = np.array([[22, 154], [1602, 170], [1630, 1066], [42, 1054]], dtype=np.int32)

people_enter_queue = {}
timespent = []

filename = "queue_time.csv"

# Open the file in write mode
file = open(filename, 'w', newline='')

# Create a CSV writer object
csv_writer = csv.writer(file)

frame_count = 0

# Create a list to store frames for output video
frames_for_output = []

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        annotated_frame = frame.copy()

        # Draw the ROIs
        cv2.drawContours(annotated_frame, [roi1_coords], -1, (255, 0, 0), 3)
        cv2.drawContours(annotated_frame, [roi2_coords], -1, (255, 0, 0), 3)

        # Run YOLOv8 tracking on the original frame
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xyxy.cpu()

        print("Found: ", people_enter_queue)

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Check if the center of bounding boxes is inside the ROI
            for box, track_id in zip(boxes, track_ids):
                print("Tracking:", track_id)

                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                x = (x1 + x2) / 2
                y = (y1 + y2) / 2

                # Visualize the people being tracked in queues on the frame
                if ((cv2.pointPolygonTest(roi1_coords, (x, y), False) > 0) or (
                        (cv2.pointPolygonTest(roi2_coords, (x, y), False)) > 0)):
                    if str(track_id) not in people_enter_queue:
                        # Get timestamp
                        people_enter_queue[str(track_id)] = str(frame_count)

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, "Person id:" + str(track_id), (x1, y1 - 5),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (20, 255, 0), 2)
                else:
                    print("outside:", track_id)

                    if str(track_id) in people_enter_queue:
                        # Get timestamp
                        exit = frame_count

                        # Get first timestamp
                        start = people_enter_queue[str(track_id)]

                        time_spent = (exit - int(start)) / fps
                        print("time spent ", time_spent, "by person", track_id)

                        timespent.append(time_spent)

                        # Write string to the file
                        csv_writer.writerow(
                            ["Time spent by person " + str(track_id) + " in line is " + str(time_spent)])

                        people_enter_queue.pop(str(track_id))

        # Add annotated frame to the list of frames for output video
        frames_for_output.append(annotated_frame)
        frame_count += 1

    else:
        # Break the loop if there are no more frames
        break

# Save frames to output video using imageio
imageio.mimwrite(output_video_path, frames_for_output, fps=fps)

# Calculate average time spent in the queue
if timespent:
    average = sum(timespent) / len(timespent)
    print("Average of list: ", round(average, 3))

    # Write average time spent to the CSV file
    csv_writer.writerow(["Average time spent in line is " + str(round(average, 3))])

# Release the video capture object, release the output video, and close the display window
cap.release()
file.close()

print(f"Output video saved at: {output_video_path}")
