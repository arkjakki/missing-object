import cv2
import numpy as np
import time
import requests
import csv
from datetime import datetime

# Function to get location from IP
def get_location_from_ip():
    response = requests.get('https://ipinfo.io/')
    data = response.json()
    location = data['loc'].split(',')  # Returns latitude and longitude
    latitude = float(location[0])
    longitude = float(location[1])
    return latitude, longitude

# Function to show popup (using OpenCV window to simulate popup)
def show_popup(message, frame, locations=None):
    cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if locations:
        ref_location, curr_location = locations
        cv2.putText(frame, f"Ref: {ref_location}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Curr: {curr_location}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.circle(frame, ref_location, 10, (0, 255, 0), -1)  # Mark reference location
        cv2.circle(frame, curr_location, 10, (0, 0, 255), -1)  # Mark current location
    cv2.imshow("Missing Object Detection", frame)

# Function to detect missing objects by comparing reference image to current frame
def detect_missing_objects(reference_image, current_frame):
    # Convert both images to grayscale
    gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors for both reference and current frames
    kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
    kp_curr, des_curr = orb.detectAndCompute(gray_curr, None)

    # Validate descriptors
    if des_ref is None or des_curr is None:
        return "Error: Unable to detect sufficient features in one or both images.", None

    # Use Brute Force Matcher to find matches between the reference and current images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    try:
        matches = bf.knnMatch(des_ref, des_curr, k=2)  # knnMatch returns list of matches
    except cv2.error as e:
        print(f"Error during knnMatch: {e}")
        return "Error: Descriptor matching failed.", None

    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:  # Ensure that we have two matches
            m, n = match_pair
            if m.distance < 0.7 * n.distance:  # Lowe's ratio test
                good_matches.append(m)

    # Check the number of good matches
    if len(good_matches) > 10:  # Adjust threshold based on your needs
        return "No changes detected", None

    # Find the location of keypoints in the reference and current images
    ref_pts = np.array([kp_ref[m.queryIdx].pt for m in good_matches])
    curr_pts = np.array([kp_curr[m.trainIdx].pt for m in good_matches])

    # Calculate the average location of keypoints
    if len(ref_pts) > 0 and len(curr_pts) > 0:
        ref_location = tuple(np.mean(ref_pts, axis=0).astype(int))
        curr_location = tuple(np.mean(curr_pts, axis=0).astype(int))
        return "Object missing detected", (ref_location, curr_location)
    else:
        return "Object missing detected", None

# Function to capture reference image using webcam
def capture_reference_image():
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("Capturing reference image...")

    ret, reference_image = cap.read()
    if not ret:
        print("Error: Failed to capture reference image.")
        return None

    cv2.imshow("Reference Image Captured", reference_image)
    cv2.waitKey(2000)  # Show the captured image for 2 seconds

    cap.release()

    return reference_image

# Function to save data to a CSV file
def save_to_csv(status, latitude, longitude):
    with open("object_detection_log.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, latitude, longitude, status])
    print(f"Data saved: {timestamp}, {latitude}, {longitude}, {status}")

# Function to start webcam and detect missing objects
def start_webcam_detection(reference_image):
    cap = cv2.VideoCapture(0)  # Open webcam again

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting to detect missing objects...")

    latitude, longitude = get_location_from_ip()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        feedback_message, location = detect_missing_objects(reference_image, frame)
        save_to_csv(feedback_message, latitude, longitude)
        show_popup(feedback_message, frame, location)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the full process in sequence
if __name__ == "__main__":
    # Create CSV file and add header if it doesn't exist
    with open("object_detection_log.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Latitude", "Longitude", "Status"])
    
    reference_image = capture_reference_image()
    if reference_image is not None:
        print("Waiting for 10 seconds before starting detection...")
        time.sleep(10)

        start_webcam_detection(reference_image)
