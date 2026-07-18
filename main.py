import cv2
from ultralytics import YOLO
import numpy as np
import urllib.request

def main():
    # Load a pre-trained YOLOv9 model.
    # The 'yolov9c.pt' model will be downloaded automatically if not found locally.
    # This model is trained on the COCO dataset and can detect 80 different object classes.
    model = YOLO('yolov9c.pt') # Load the YOLOv9 model

    # Define the URL for a sample image.
    # This image will be downloaded and used for object detection.
    image_url = "https://ultralytics.com/images/bus.jpg" # Sample image URL

    print(f"Downloading image from: {image_url}")
    try:
        # Download the image from the URL
        req = urllib.request.urlopen(image_url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1) # Decode image into memory
        if img is None:
            raise ValueError("Could not decode image from URL.")
        print("Image downloaded successfully.")
    except Exception as e:
        print(f"Error downloading or decoding image: {e}")
        print("Please ensure you have an active internet connection and the URL is valid.")
        return

    # Perform object detection on the image using the loaded YOLOv9 model.
    # The 'conf' parameter sets the object confidence threshold (e.g., 0.25 means 25% confidence).
    # The 'iou' parameter sets the Intersection Over Union threshold for non-maximum suppression.
    results = model(img, conf=0.25, iou=0.7) # Perform object detection

    # Process and display the detection results.
    # Each 'result' object contains bounding boxes, confidence scores, and class labels.
    for r in results:
        # Get the annotated image with bounding boxes and labels drawn.
        # This uses Ultralytics' built-in plotting functionality to visualize detections.
        annotated_img = r.plot() # Draw detected objects on the image

        # Display the annotated image in a window.
        # 'YOLOv9 Object Detection' will be the window title.
        cv2.imshow("YOLOv9 Object Detection", annotated_img) # Show the results
        cv2.waitKey(0) # Wait for a key press to close the window

    cv2.destroyAllWindows() # Close all OpenCV windows

if __name__ == "__main__":
    main()
