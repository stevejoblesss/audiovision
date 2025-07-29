import speech_recognition as sr
import threading
import cv2
import numpy as np
import pyttsx3
import queue
import time
import serial
import pynmea2
import requests
import openrouteservice

# === CONFIG ===
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjRlZDhhMjFmNTk5YjQxNmE4OTdlZjA2YTNjMTBhZWEyIiwiaCI6Im11cm11cjY0In0="
DESTINATION_HOME = ("Chung Ling Private High School, Lrg Kg Baru, 11400 Ayer Itam, Pulau Pinang")

# === TTS ===
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 125)
speech_queue = queue.Queue()


def speak_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        tts_engine.say(text)
        tts_engine.runAndWait()
        speech_queue.task_done()


threading.Thread(target=speak_worker, daemon=True).start()


def speak(text):
    print("🔊", text)
    speech_queue.put(text)


# === GPS ===
def get_gps_location():
    gps = serial.Serial("/dev/ttyAMA0", baudrate=9600, timeout=1)
    while True:
        try:
            line = gps.readline().decode("ascii", errors="replace")
            if line.startswith("$GPGGA") or line.startswith("$GPRMC"):
                msg = pynmea2.parse(line)
                if msg.latitude and msg.longitude:
                    return msg.latitude, msg.longitude
        except:
            pass
        time.sleep(0.5)


# === ORS Navigation ===
def get_coordinates_from_address(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json"}
    response = requests.get(url, params=params).json()
    if response:
        return float(response[0]["lat"]), float(response[0]["lon"])
    return None, None


def get_directions(lat, lon, destination_address):
    client = openrouteservice.Client(key=ORS_API_KEY)
    dest_lat, dest_lon = get_coordinates_from_address(destination_address)
    if dest_lat is None:
        return []
    coords = ((lon, lat), (dest_lon, dest_lat))
    try:
        routes = client.directions(coords, profile="foot-walking")
        return routes["routes"][0]["segments"][0]["steps"]
    except:
        return []


def get_address(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json"}
    response = requests.get(url, params=params).json()
    return response.get("display_name", "Unknown location")


def search_place_nearby(lat, lon, keyword):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": keyword, "format": "json", "limit": 1, "lat": lat, "lon": lon}
    response = requests.get(url, params=params).json()
    return response[0]["display_name"] if response else None


# === Speech Recognition ===
def listen_for_hotword():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        speak("Say 'Hey Avis' to activate me.")
        while True:
            try:
                audio = r.listen(source, timeout=5)
                phrase = r.recognize_google(audio).lower()
                if "hey avis" in phrase:     # hot word part
                    speak("Yes, I'm listening.")
                    return listen_for_command(r, source)
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue


def listen_for_command(r, source):
    speak("Please say a command.")
    try:
        audio = r.listen(source, timeout=6)
        cmd = r.recognize_google(audio).lower()
        print("Command:", cmd)
        return cmd
    except:
        speak("I didn't catch that.")
        return None


# === Object Detection (with stairs) ===
def object_detection_thread():
    KNOWN_WIDTH = 55
    KNOWN_DISTANCE = 130
    REF_OBJECT_PIXEL_WIDTH = 325
    FOCAL_LENGTH = (REF_OBJECT_PIXEL_WIDTH * KNOWN_DISTANCE) / KNOWN_WIDTH
    last_announcement_time = 0
    cooldown = 5

    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    stairs_net = cv2.dnn.readNet(
        "stairs-yolov3-tiny_6500.weights", "stairs-yolov3-tiny.cfg"
    )
    CLASSES = open("coco.names").read().strip().split("\n")
    STAIRS_CLASSES = open("stairs.names").read().strip().split("\n")

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("AudioVision", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AudioVision", 960, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (256, 256), swapRB=True, crop=False
        )
        net.setInput(blob)
        detections = net.forward(net.getUnconnectedOutLayersNames())

        closest_object, closest_box, min_steps = None, None, float("inf")

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.4:
                    box = obj[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, box_width, box_height) = box.astype("int")
                    startX = int(centerX - box_width / 2)
                    startY = int(centerY - box_height / 2)
                    endX = startX + box_width
                    endY = startY + box_height
                    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / box_width
                    steps = max(1, int(round(distance / 50)))
                    label = f"{CLASSES[class_id]}: {steps} steps"
                    if steps < min_steps:
                        closest_object = label
                        closest_box = (startX, startY, endX, endY)
                        min_steps = steps
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        # Announce closest object
        if closest_box and time.time() - last_announcement_time > cooldown:
            (startX, startY, endX, endY) = closest_box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(
                frame,
                closest_object,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            speak(closest_object)
            last_announcement_time = time.time()

        # --- STAIRS DETECTION ---
        stairs_blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (256, 256), swapRB=True, crop=False
        )
        stairs_net.setInput(stairs_blob)
        stairs_detections = stairs_net.forward(
            stairs_net.getUnconnectedOutLayersNames()
        )
        for detection in stairs_detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.4:
                    box = obj[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, box_width, box_height) = box.astype("int")
                    startX = int(centerX - box_width / 2)
                    startY = int(centerY - box_height / 2)
                    endX = startX + box_width
                    endY = startY + box_height
                    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / box_width
                    steps = max(1, int(round(distance / 50)))
                    label = f"{STAIRS_CLASSES[class_id]}: {steps} steps"
                    if time.time() - last_announcement_time > cooldown:
                        speak(label)
                        last_announcement_time = time.time()
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

        cv2.imshow("AudioVision", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)


# === MAIN ===
def main():
    threading.Thread(target=object_detection_thread, daemon=True).start()
    while True:
        cmd = listen_for_hotword()
        if cmd:
            if "take me home" in cmd:
                speak("Planning route to home.")
                lat, lon = get_gps_location()
                print("Current lat/lon:", lat, lon)
                steps = get_directions(lat, lon, DESTINATION_HOME)
                for i, step in enumerate(steps):
                    speak(f"Step {i+1}: {step['instruction']}")

            elif "where am i" in cmd:
                lat, lon = get_gps_location()
                print("Current lat/lon:", lat, lon)
                speak(f"You are near {get_address(lat, lon)}")

            elif "nearest clinic" in cmd:
                lat, lon = get_gps_location()
                print("Current lat/lon:", lat, lon)
                place = search_place_nearby(lat, lon, "clinic")
                speak(f"Nearest clinic is at {place}" if place else "Clinic not found")
                
            else:
                print("Command not recognised")
                speak("Command not recognised")


if __name__ == "__main__":
    main()
