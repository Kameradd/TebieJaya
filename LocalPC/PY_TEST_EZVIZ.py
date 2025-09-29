# file: test_camera.py

import cv2

# --- CONFIGURATION ---
# IMPORTANT: Replace "YOUR_CODE" with the 6-character verification code
#            printed on the sticker on your camera.
VERIFICATION_CODE = "ILBUCZ"
IP_ADDRESS = "192.168.10.64" # The new IP address you set

# Construct the full RTSP URL
RTSP_URL = f"rtsp://admin:{VERIFICATION_CODE}@{IP_ADDRESS}:554"

# --- MAIN SCRIPT ---
print(f"Connecting to camera stream at: {RTSP_URL}")

# Connect to the video stream
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

# Check if the connection was successful
if not cap.isOpened():
    print("❌ Error: Could not open camera stream.")
    print("Please check the following:")
    print("1. Is the camera powered on and connected?")
    print("2. Is the RTSP_URL correct? (Check IP address and verification code)")
    print("3. Is your PC on the same network (192.168.10.x)?")
    exit()

print("✅ Stream opened successfully! A window should appear.")
print("Press 'q' in the video window to quit.")

# Loop to read frames from the camera and display them
while True:
    # Read one frame from the video stream
    ret, frame = cap.read()
    
    # If a frame was not received, it might be the end of the stream or an error
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break
        
    # Display the resulting frame in a window
    cv2.imshow('Ezviz Camera Stream', frame)
    
    # Wait for 1ms and check if the 'q' key was pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
print("Closing stream and cleaning up...")
cap.release()
cv2.destroyAllWindows()
