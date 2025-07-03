import cv2
import numpy as np
import pyttsx3
import threading
import queue
import time

# Reference object constants for focal length
KNOWN_WIDTH = 55  # cm
KNOWN_DISTANCE = 130  # cm
REF_OBJECT_PIXEL_WIDTH = 325
FOCAL_LENGTH = (REF_OBJECT_PIXEL_WIDTH * KNOWN_DISTANCE) / KNOWN_WIDTH

# Configurable parameters
MAX_STEPS_TO_ANNOUNCE = 15
SIDE_BOUNDARY_PERCENT = 0.33
clrTime = 4  # Clear speech queue interval
ALLOWED_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorbike",
    "bus",
    "truck",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "bench",
    "dog",
    "horse",
    "sheep",
    "cow",
    "bear",
    "suitcase",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "sink",
    "refrigerator",
    "vase",
}

# Load general YOLO model
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open("coco.names", "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Load stairs YOLO model
stairs_net = cv2.dnn.readNet(
    "stairs-yolov3-tiny_6500.weights", "stairs-yolov3-tiny.cfg"
)
stairs_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
stairs_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open("stairs.names", "r") as f:
    STAIRS_CLASSES = [line.strip() for line in f.readlines()]

# TTS setup
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 125)
tts_engine.setProperty("volume", 1.0)

speech_queue = queue.Queue()
last_announced = {}
queue_clear_time = time.time()
periodic_message = "Stay aware of your surroundings."
periodic_message_interval = 30
last_periodic_time = time.time()


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
    return (
        (KNOWN_WIDTH * FOCAL_LENGTH) / object_width_in_frame
        if object_width_in_frame > 0
        else float("inf")
    )


# Start video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width = frame.shape[:2]
    left_boundary = width * SIDE_BOUNDARY_PERCENT
    right_boundary = width * (1 - SIDE_BOUNDARY_PERCENT)

    # --- General YOLO detection ---
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (256, 256), swapRB=True, crop=False)
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
    closest_object, closest_distance, closest_label, closest_box = (
        None,
        float("inf"),
        "",
        None,
    )

    if len(indices) > 0:
        for i in indices.flatten():
            startX, startY, box_width, box_height = boxes[i]
            class_id = class_ids[i]
            endX = startX + box_width
            endY = startY + box_height
            distance = calculate_distance(box_width)
            steps = max(1, int(round(distance / 50, 0)))
            if steps > MAX_STEPS_TO_ANNOUNCE:
                continue
            object_center = startX + (box_width // 2)
            position = (
                "left"
                if object_center < left_boundary
                else "right" if object_center > right_boundary else "ahead"
            )
            label = f"{CLASSES[class_id]}: {steps} steps {distance:.2f}cm ({position})"
            object_key = (CLASSES[class_id], position)
            last_steps = last_announced.get(object_key, None)
            if last_steps is None or abs(last_steps - steps) >= 1:
                closest_distance = distance
                closest_object = label
                closest_label = label
                closest_box = (startX, startY, endX, endY)
                last_announced[object_key] = steps

    # --- Stairs Detection ---
    stairs_blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (256, 256), swapRB=True, crop=False
    )
    stairs_net.setInput(stairs_blob)
    stairs_detections = stairs_net.forward(stairs_net.getUnconnectedOutLayersNames())

    stairs_boxes, stairs_confidences, stairs_class_ids = [], [], []

    for detection in stairs_detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                box = obj[0:4] * np.array([width, height, width, height])
                (centerX, centerY, box_width, box_height) = box.astype("int")
                startX = int(centerX - (box_width / 2))
                startY = int(centerY - (box_height / 2))
                stairs_boxes.append([startX, startY, box_width, box_height])
                stairs_confidences.append(float(confidence))
                stairs_class_ids.append(class_id)

    stairs_indices = cv2.dnn.NMSBoxes(stairs_boxes, stairs_confidences, 0.4, 0.3)

    if len(stairs_indices) > 0:
        for i in stairs_indices.flatten():
            startX, startY, box_width, box_height = stairs_boxes[i]
            endX = startX + box_width
            endY = startY + box_height
            object_width_in_frame = box_width
            distance = calculate_distance(object_width_in_frame)
            steps = max(1, int(round(distance / 50, 0)))
            position = "ahead"  # or add L/R logic
            label = f"{STAIRS_CLASSES[stairs_class_ids[i]]}: {steps} steps {distance:.2f}cm ({position})"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            cv2.putText(
                frame,
                label,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
            speak(label)

    # Speak closest object from general model
    if closest_object:
        print("Announcing:", closest_label)
        speak(closest_label)

    # Draw general detections
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
            cv2.putText(
                frame,
                label,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    # Periodic reminder
    if time.time() - last_periodic_time >= periodic_message_interval:
        speak(periodic_message)
        last_periodic_time = time.time()

    # Clear speech queue
    if time.time() - queue_clear_time > clrTime:
        with speech_queue.mutex:
            speech_queue.queue.clear()
        queue_clear_time = time.time()

    cv2.imshow("AudioVision", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
speech_queue.put(None)
speech_thread.join()
cap.release()
cv2.destroyAllWindows()
