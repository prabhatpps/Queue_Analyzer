# Queue Analyzer

Queue Analyzer is an advanced computer vision project designed to revolutionize queue management by providing real-time monitoring and analysis capabilities. Leveraging state-of-the-art YOLOv8 object detection, Queue Analyzer accurately tracks individuals within predefined regions of interest, offering insights into queue lengths, wait times, and service rates. This innovative solution empowers businesses across diverse industries to optimize customer experiences, streamline operations, and drive operational excellence.

## Features

- **Real-time Queue Monitoring:** Utilizes YOLOv8 object detection for accurate and real-time monitoring of individuals within defined regions of interest.
- **Queue Analysis:** Provides insights into queue lengths, wait times, and service rates, enabling businesses to optimize customer experiences and streamline operations.
- **Customizable Parameters:** Offers flexibility with customizable parameters for defining regions of interest, tracking thresholds, and visualization options.
- **Data Export:** Facilitates data export in various formats for further analysis and integration with existing systems.
- **Scalability:** Scalable architecture to accommodate varying queue sizes and complexities in different environments.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/queue-analyzer.git
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
in case YOLO is not getting installed --- Install ultralytics:
```bash
pip install ultralytics
```
## Usage

1. Configure the parameters.
2. Run the main script:

```bash
python queue_analyzer.py
```

And here's the Configuration section with details on configuring parameters:

## Configuration

- **ROI Coordinates:** Define the coordinates of the regions of interest (ROI) where queues are to be monitored.
- **Thresholds:** Set thresholds for object detection and tracking parameters.
- **Output Settings:** Configure output options such as video output format and file paths.
- **Logging:** Enable logging and specify log file paths for recording events and errors.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.

## Acknowledgements

- YOLOv8 object detection model by Ultralytics: [Link](https://github.com/ultralytics/yolov5)
- OpenCV: [Link](https://github.com/opencv/opencv)
- NumPy: [Link](https://github.com/numpy/numpy)

## Contact

For questions or inquiries, please contact Prabhat Pandey.
