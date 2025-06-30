import cv2
import numpy as np
import pyttsx3
import threading
import queue
import time

# Reference Object Constants
KNOWN_WIDTH = 55  # cm
KNOWN_DISTANCE = 130  # cm
REF_OBJECT_PIXEL_WIDTH = 325  # Measure manually for accuracy

# Compute focal length
FOCAL_LENGTH = (REF_OBJECT_PIXEL_WIDTH * KNOWN_DISTANCE) / KNOWN_WIDTH

# Adjustable Parameters
MAX_STEPS_TO_ANNOUNCE = 15  # Objects beyond this step count won't be announced
SIDE_BOUNDARY_PERCENT = 0.33  # Percentage of screen width for left/right detection
clrTime = 4 # amount of time to clear queue   default:4
ALLOWED_CLASSES = {"stair",}  # Add the object classes you want to detect

# Load YOLO
net = cv2.dnn.readNet("stairs-yolov3-tiny_6500.weights", "stairs-yolov3-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class labels
with open("stairs.names", "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Initialize TTS Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 1.0)

speech_queue = queue.Queue()
last_announced = {}  # Stores last known (object, steps, position)
queue_clear_time = time.time()  # Tracks when to clear speech queue

# Speech worker thread
def speak_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        tts_engine.say(text)
        tts_engine.runAndWait()
        speech_queue.task_done()

speech_thread = threading.Thread(target=speak_worker, daemon=True)
speech_thread.start()

def speak(text):
    speech_queue.put(text)

def calculate_distance(object_width_in_frame):
    return (KNOWN_WIDTH * FOCAL_LENGTH) / object_width_in_frame if object_width_in_frame > 0 else float('inf')

# Start Video Capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width = frame.shape[:2]
    left_boundary = width * SIDE_BOUNDARY_PERCENT
    right_boundary = width * (1 - SIDE_BOUNDARY_PERCENT)

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (256, 256), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(net.getUnconnectedOutLayersNames())

    boxes, confidences, class_ids = [], [], []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            class_name = CLASSES[class_id]
            if confidence > 0.4 and class_name in ALLOWED_CLASSES:
                box = obj[0:4] * np.array([width, height, width, height])
                (centerX, centerY, box_width, box_height) = box.astype("int")
                startX = int(centerX - (box_width / 2))
                startY = int(centerY - (box_height / 2))

                boxes.append([startX, startY, box_width, box_height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    closest_object, closest_distance, closest_label, closest_box = None, float('inf'), "", None

    if len(indices) > 0:
        for i in indices.flatten():
            startX, startY, box_width, box_height = boxes[i]
            class_id = class_ids[i]

            endX = startX + box_width
            endY = startY + box_height
            object_width_in_frame = box_width

            if object_width_in_frame > 0:
                distance = calculate_distance(object_width_in_frame)
                steps = max(1, int(round(distance / 50, 0)))

                if steps > MAX_STEPS_TO_ANNOUNCE:
                    continue  # Skip objects beyond max steps

                object_center = startX + (box_width // 2)
                position = "left" if object_center < left_boundary else "right" if object_center > right_boundary else "ahead"

                label = f"{CLASSES[class_id]}: {steps} steps {distance:.2f}cm ({position})"
                
                object_key = (CLASSES[class_id], position)
                last_steps = last_announced.get(object_key, None)

                if last_steps is None or abs(last_steps - steps) >= 1:
                    closest_distance = distance
                    closest_object = f"{CLASSES[class_id]}, {steps} steps, {position}"
                    closest_label = f"{CLASSES[class_id]} {steps} steps {position}"
                    closest_box = (startX, startY, endX, endY)
                    last_announced[object_key] = steps

    if closest_object:
        print("Announcing:", closest_label)
        speak(closest_label)

    # Draw all boxes
    if len(indices) > 0:
        for i in indices.flatten():
            startX, startY, box_width, box_height = boxes[i]
            endX = startX + box_width
            endY = startY + box_height
            color = (0, 255, 0)
            if closest_box and (startX, startY, endX, endY) == closest_box:
                color = (0, 0, 255)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            label = f"{CLASSES[class_ids[i]]}: {max(1, int(round(calculate_distance(box_width) / 50, 0)))} steps"
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Clear the speech queue every 5 seconds
    if time.time() - queue_clear_time > clrTime:
        with speech_queue.mutex:
            speech_queue.queue.clear()
        queue_clear_time = time.time()

    cv2.imshow("Object Detection and Distance Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

speech_queue.put(None)
speech_thread.join()
cap.release()
cv2.destroyAllWindows()

