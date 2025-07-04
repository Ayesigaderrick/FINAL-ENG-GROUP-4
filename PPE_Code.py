import cv2
from picamera2 import Picamera2
import numpy as np
from ultralytics import YOLO
import serial
import time
import os
from datetime import datetime
import RPi.GPIO as GPIO

# Configuration
PHONE_NUMBER = "+256778724647"  

# Initialize SIM800L
def init_sim800l(port="/dev/serial0", baudrate=9600):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        ser.write(b'AT\r\n')
        response = ser.read(100).decode()
        if "OK" in response:
            print("SIM800L initialized successfully on /dev/serial0 at 9600 baud")
            return ser
        else:
            print("SIM800L initialization failed")
            return None
    except Exception as e:
        print(f"Error initializing SIM800L: {e}")
        return None

# Function to make a call and hang up after 10 seconds
def make_call(ser, phone_number):
    try:
        ser.write(b'ATD' + phone_number.encode() + b';\r\n')
        time.sleep(1)
        response = ser.read(100).decode()
        if "OK" in response:
            print(f"Calling {phone_number}")
            time.sleep(10)  # Call duration: 10 seconds
            ser.write(b'ATH\r\n')  # Hang up
            time.sleep(1)
            hangup_response = ser.read(100).decode()
            if "OK" in hangup_response:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Call ended successfully at {timestamp}")
            else:
                print("Failed to end call")
        else:
            print("Failed to initiate call")
    except Exception as e:
        print(f"Error during call: {e}")

# Function to send an SMS with a timestamp
def send_sms(ser, phone_number, message):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        ser.write(b'AT+CMGF=1\r\n')  # Set SMS to text mode
        time.sleep(0.5)
        response = ser.read(100).decode()
        if "OK" not in response:
            print("Failed to set SMS text mode")
            return
        ser.write(b'AT+CMGS="' + phone_number.encode() + b'"\r\n')
        time.sleep(0.5)
        ser.write(full_message.encode() + b'\r\n')
        time.sleep(0.5)
        ser.write(bytes([26]))  # Ctrl+Z to send
        time.sleep(1)
        response = ser.read(100).decode()
        if "OK" in response:
            print(f"SMS sent to {phone_number}: {full_message}")
        else:
            print("Failed to send SMS")
    except Exception as e:
        print(f"Error sending SMS: {e}")

# Function to trigger buzzer beeps
def trigger_buzzer(pin):
    for _ in range(4):  # Beep 4 times
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(1)  # Beep for 0.1 seconds
        GPIO.output(pin, GPIO.LOW)
        time.sleep(0.5)  # Pause for 0.1 seconds

# Load the YOLOv11 model
model = YOLO("PPE_FINAL.pt")

# Define class names
class_names = ["Helment", "No Safety_Vest", "No_Helmet", "Person", "Safety_Vest"]

# Define colors for non-person classes (BGR format)
colors = {
    "Helment": (255, 255, 0),       # Cyan
    "No Safety_Vest": (0, 255, 255), # Yellow
    "No_Helmet": (255, 255, 255),   # White
    "Person": (0, 0, 255),          # Default red, overridden by compliance
    "Safety_Vest": (238, 130, 238)  # Violet
}

# Function to check if a point is within a bounding box
def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

# Create logs directory for call-triggered images
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Violation tracking variables
violation_count = 0
can_call = True
cooldown_counter = 0
VIOLATION_THRESHOLD = 10
COOLDOWN_FRAMES = 10
frame_count = 0

# Initialize GPIO for buzzer
BUZZER_PIN = 18  # GPIO 18 (physical pin 12)
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)  # Ensure buzzer is off initially

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
picam2.start()
print("Camera initialized and ready")

# Initialize SIM800L
ser = init_sim800l()

# Function to print status with frame countdown
def print_status(frame_count, num_non_compliant, can_call, violation_count, cooldown_counter, VIOLATION_THRESHOLD):
    if num_non_compliant > 0:
        if can_call:
            message = f"Frame: {frame_count}, Non-compliant: {num_non_compliant}, Violation count: {violation_count}/{VIOLATION_THRESHOLD}"
        else:
            message = f"Frame: {frame_count}, Non-compliant: {num_non_compliant}, Cooldown: {cooldown_counter} frames remaining"
    else:
        message = f"Frame: {frame_count}, No violations detected"
    message = message.ljust(80)
    print(message, end='\r')

try:
    while True:
        frame_count += 1
        start_time = time.time()
        frame = picam2.capture_array()
        
        # Run inference
        results = model(frame, conf=0.5)
        boxes = results[0].boxes
        
        # Extract helmets and vests
        helmets = [box for box in boxes if class_names[int(box.cls)] == "Helment"]
        vests = [box for box in boxes if class_names[int(box.cls)] == "Safety_Vest"]
        
        num_persons = 0
        num_non_compliant = 0
        
        # Process each detection
        for box in boxes:
            cls = int(box.cls)
            label = class_names[cls]
            
            if label == "Person":
                num_persons += 1
                person_box = box.xyxy[0]
                has_helmet = any(point_in_box(((h.xyxy[0][0] + h.xyxy[0][2])/2, 
                                              (h.xyxy[0][1] + h.xyxy[0][3])/2), 
                                              person_box) for h in helmets)
                has_vest = any(point_in_box(((v.xyxy[0][0] + v.xyxy[0][2])/2, 
                                            (v.xyxy[0][1] + v.xyxy[0][3])/2), 
                                            person_box) for v in vests)
                
                if has_helmet and has_vest:
                    color = (0, 255, 0)  # Green for compliant
                else:
                    color = (0, 0, 255)  # Red for violation
                    num_non_compliant += 1
                
                x1, y1, x2, y2 = map(int, person_box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                color = colors[label]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label}: {box.conf[0]:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Handle violation tracking and call triggering
        if num_non_compliant > 0:
            if can_call:
                violation_count += 1
                if violation_count >= VIOLATION_THRESHOLD and ser:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    call_filename = os.path.join(log_dir, f"call_triggered_{timestamp.replace(' ', '_').replace(':', '-')}.jpg")
                    cv2.imwrite(call_filename, frame)
                    print(f"\nImage saved: {call_filename}")
                    print(f"Initiating alert sequence at {timestamp}")
                    trigger_buzzer(BUZZER_PIN)  # Beep buzzer first (4 times, 0.1s on/off)
                    make_call(ser, PHONE_NUMBER)
                    send_sms(ser, PHONE_NUMBER, "PPE Violation Detected")
                    violation_count = 0
                    can_call = False
                    cooldown_counter = COOLDOWN_FRAMES
        else:
            violation_count = 0
            can_call = True
            GPIO.output(BUZZER_PIN, GPIO.LOW)  # Ensure buzzer is off when compliant
        
        if cooldown_counter > 0:
            cooldown_counter -= 1
            if cooldown_counter == 0:
                can_call = True
        
        print_status(frame_count, num_non_compliant, can_call, violation_count, cooldown_counter, VIOLATION_THRESHOLD)
        
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("PPE Detection", frame)
        
        if frame_count % 10 == 0:
            num_compliant = num_persons - num_non_compliant
            print(f"\nFPS: {fps:.1f}, Persons: {num_persons} (Compliant: {num_compliant}, Non-compliant: {num_non_compliant})")
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nScript interrupted by user")
finally:
    if ser:
        ser.close()
    GPIO.cleanup()
    picam2.stop()
    cv2.destroyAllWindows()
