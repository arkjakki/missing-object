# import pygame

# # Function to play the alarm sound using pygame
# def play_voice_alert():
#     try:
#         pygame.mixer.init()  # Initialize pygame mixer
#         pygame.mixer.music.load('object_missing.mp3')  # Load the MP3 file
#         pygame.mixer.music.play()  # Play the sound
#     except Exception as e:
#         print(f"Error playing sound: {e}")

# import cv2
# import numpy as np
# import time
# import threading
# from playsound import playsound

# # Function to play the alarm sound
# def play_voice_alert():
#     try:
#         playsound('object_missing.mp3')  # Make sure 'object_missing.mp3' is in the same directory
#     except:
#         print("Error playing sound!")

# # Function to show popup (using OpenCV window to simulate popup)
# def show_popup(message, frame):
#     cv2.putText(frame, message, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     cv2.imshow("Missing Object Detection", frame)

# # Function to detect missing objects by comparing reference image to current frame
# def detect_missing_objects(reference_image, current_frame):
#     # Convert both images to grayscale
#     gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
#     gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
# # Initialize ORB detector
#     orb = cv2.ORB_create()

#     # Detect keypoints and descriptors for both reference and current frames
#     kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
#     kp_curr, des_curr = orb.detectAndCompute(gray_curr, None)

#     # Use Brute Force Matcher to find matches between the reference and current images
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
#     matches = bf.knnMatch(des_ref, des_curr, k=2)  # knnMatch returns list of matches

#     # Apply Lowe's ratio test
#     good_matches = []
#     for match_pair in matches:
#         if len(match_pair) == 2:  # Ensure that we have two matches
#             m, n = match_pair
#             if m.distance < 0.7 * n.distance:  # Lowe's ratio test
#                 good_matches.append(m)

#     # Check the number of good matches and determine if any object is missing
#     if len(good_matches) > 10:  # Adjust threshold based on your needs
#         return "No changes detected"
#     else:
#         return "Object missing detected"

# # Function to capture reference image using webcam
# def capture_reference_image():
#     cap = cv2.VideoCapture(0)  # Open webcam

#     # Check if webcam is opened successfully
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return None

#     print("Capturing reference image...")

# # Capture reference image
#     ret, reference_image = cap.read()
#     if not ret:
#         print("Error: Failed to capture reference image.")
#         return None

#     # Display reference image to user
#     cv2.imshow("Reference Image Captured", reference_image)
#     cv2.waitKey(2000)  # Show the captured image for 2 seconds

#     # Release the webcam
#     cap.release()

#     return reference_image

# # Function to start webcam and detect missing objects
# def start_webcam_detection(reference_image):
#     cap = cv2.VideoCapture(0)  # Open webcam again

#     # Check if webcam is opened successfully
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     print("Starting to detect missing objects...")

#     # Start the monitoring process
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture frame.")
#             break

#         # Detect missing objects by comparing with the reference image
#         feedback_message = detect_missing_objects(reference_image, frame)
#         show_popup(feedback_message, frame)
#  # If an object is missing, play a sound alert
#         if feedback_message == "Object missing detected":
#             play_voice_alert()
#             break

#         # Break the loop if the 'q' key is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the webcam and close all OpenCV windows
#     cap.release()
#     cv2.destroyAllWindows()

# # Run the full process in sequence
# if __name__ == "__main__":
#     # Step 1: Capture the reference image
#     reference_image = capture_reference_image()
#     if reference_image is not None:
#         print("Waiting for 10 seconds before starting detection...")
#         time.sleep(10)  # Wait for 10 seconds before starting detection

#         # Step 2: Start detecting missing objects with the reference image
#         start_webcam_detection(reference_image)
from playsound import playsound

def play_voice_alert():
    try:
        playsound('object_missing.mp3')  # Play the sound
    except Exception as e:
        print(f"Error playing sound: {e}")

