import cv2
from ultralytics import YOLO

# 1. Load the YOLO model
model = YOLO("yolov8s.pt")

# 2. Open the default camera
# (0 is usually the built-in camera or the first USB camera plugged into the Pi)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

print("Camera started. Press 'q' in the video window to quit.")

# 3. Start a continuous loop to capture frames live
while True:
    # Read a single frame from the camera
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    # 4. Run inference on the current frame
    # stream=True is highly recommended for live video to keep memory usage low
    results = model(frame, stream=True, verbose=False)

    # 5. Process the results
    for result in results:

        # --- OPTIONAL: Your original extraction code ---
        # You can keep this if you want to trigger physical actions
        # (like turning on an LED or sending an alert) based on what is detected.
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            # print(f"Detected {label} ({confidence:.0%})")
        # -----------------------------------------------

        # Automatically draw the bounding boxes and labels onto the image frame
        annotated_frame = result.plot()

        # Display the image on your monitor in a window named "YOLO Live"
        cv2.imshow("YOLO Live", annotated_frame)

    # 6. Listen for the 'q' key to break the loop and exit
    # cv2.waitKey(1) waits 1 millisecond for a key press between frames
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

# 7. Clean up the camera and close all windows
cap.release()
cv2.destroyAllWindows()