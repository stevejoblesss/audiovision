# AudioVision All-in-One Script (with Updated Detection)

import os, re, time, json, queue, serial, pynmea2, requests, pyttsx3, vosk, sounddevice as sd, pickle, cv2, numpy as np, threading
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# === CONFIG ===
CREDENTIALS_FILE = "avis_credentials.json"
TOKEN_FILE = "token.pkl"
SCOPES = ["https://www.googleapis.com/auth/mapsplatform.directions"]
GOOGLE_MAPS_API_KEY = "AIzaSyBPZ3JcOxQtnZZhuPZhYNaB0xXjESAH8Hk"
DESTINATION_HOME = "Komtar, George Town, Penang"

# === TTS ===
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 125)
tts_engine.setProperty("volume", 1.0)
speech_queue = queue.Queue()
queue_clear_time = time.time()


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
    speech_queue.put(text)


# === VOSK ===
model = vosk.Model("vosk-model-small-en-us-0.15")
audio_q = queue.Queue()


def audio_callback(indata, frames, time_, status):
    if status:
        print(status)
    audio_q.put(bytes(indata))


def listen_command_with_hotword():
    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        recognizer = vosk.KaldiRecognizer(model, 16000)
        while True:
            print("🎤 Waiting for 'hey avis'...")
            while True:
                data = audio_q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    if "hey avis" in result.get("text", "").lower():
                        speak("Yes, I'm listening.")
                        break
            print("🎤 Awaiting command...")
            start = sd.time()
            while sd.time() - start < 7:
                data = audio_q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    cmd = result.get("text", "").lower()
                    if cmd:
                        print("Command:", cmd)
                        return cmd
            speak("I didn't catch that. Please try again.")


# === GPS / Maps ===
def get_gps_location():
    gps = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=1)
    while True:
        try:
            line = gps.readline().decode("ascii", errors="replace")
            if line.startswith("$GPGGA") or line.startswith("$GPRMC"):
                msg = pynmea2.parse(line)
                if msg.latitude and msg.longitude:
                    return msg.latitude, msg.longitude
        except Exception:
            pass
        time.sleep(0.5)


def authenticate_google():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)


def get_directions(lat, lon, destination):
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{lat},{lon}",
        "destination": destination,
        "mode": "walking",
        "key": GOOGLE_MAPS_API_KEY,
    }
    response = requests.get(url, params=params).json()
    return response.get("routes", [{}])[0].get("legs", [{}])[0].get("steps", [])


def get_address(lat, lon):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"latlng": f"{lat},{lon}", "key": GOOGLE_MAPS_API_KEY}
    response = requests.get(url, params=params).json()
    results = response.get("results", [])
    return results[0]["formatted_address"] if results else "Unknown location"


def search_place_nearby(lat, lon, keyword):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lon}",
        "radius": 2000,
        "keyword": keyword,
        "key": GOOGLE_MAPS_API_KEY,
    }
    response = requests.get(url, params=params).json()
    results = response.get("results", [])
    return results[0]["vicinity"] if results else None


# === Vision Detection ===
def object_detection_thread():
    KNOWN_WIDTH = 55
    KNOWN_DISTANCE = 130
    REF_OBJECT_PIXEL_WIDTH = 325
    FOCAL_LENGTH = (REF_OBJECT_PIXEL_WIDTH * KNOWN_DISTANCE) / KNOWN_WIDTH
    MAX_STEPS_TO_ANNOUNCE = 15
    SIDE_BOUNDARY_PERCENT = 0.33
    periodic_message = "Stay aware of your surroundings."
    periodic_message_interval = 30
    last_periodic_time = time.time()

    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    stairs_net = cv2.dnn.readNet(
        "stairs-yolov3-tiny_6500.weights", "stairs-yolov3-tiny.cfg"
    )
    CLASSES = open("coco.names").read().strip().split("\n")
    STAIRS_CLASSES = open("stairs.names").read().strip().split("\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        height, width = frame.shape[:2]
        left_boundary = width * SIDE_BOUNDARY_PERCENT
        right_boundary = width * (1 - SIDE_BOUNDARY_PERCENT)

        # General YOLO detection
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (256, 256), swapRB=True, crop=False
        )
        net.setInput(blob)
        detections = net.forward(net.getUnconnectedOutLayersNames())

        boxes, confidences, class_ids = [], [], []
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.4:
                    box = obj[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, box_width, box_height) = box.astype("int")
                    startX = int(centerX - (box_width / 2))
                    startY = int(centerY - (box_height / 2))
                    boxes.append([startX, startY, box_width, box_height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

        closest_object = None
        closest_box = None
        min_steps = float('inf')

        if len(indices) > 0:
            for i in indices.flatten():
                startX, startY, box_width, box_height = boxes[i]
                class_id = class_ids[i]
                endX = startX + box_width
                endY = startY + box_height
                distance = (KNOWN_WIDTH * FOCAL_LENGTH) / box_width
                steps = max(1, int(round(distance / 50)))

                if steps < min_steps:
                    min_steps = steps
                    closest_object = CLASSES[class_id]
                    closest_box = (startX, startY, endX, endY)

        # Draw green box with label
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        label = f"{CLASSES[class_id]}"
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2,)

        # Draw red box and speak for closest object
        if closest_box:
            print(f"Announcing: {closest_object}")
            speak(f"{closest_object} ahead in {min_steps} steps")
            (startX, startY, endX, endY) = closest_box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            label = f"{closest_object}"
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Periodic reminder
        if time.time() - last_periodic_time >= periodic_message_interval:
            speak(periodic_message)
            last_periodic_time = time.time()

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
        cmd = listen_command_with_hotword()
        if "take me home" in cmd or "go home" in cmd:
            speak("Planning route to home.")
            lat, lon = get_gps_location()
            authenticate_google()
            steps = get_directions(lat, lon, DESTINATION_HOME)
            for i, step in enumerate(steps):
                clean = re.sub("<[^<]+?>", "", step["html_instructions"])
                speak(f"Step {i+1}: {clean}")
        elif "where am i" in cmd:
            lat, lon = get_gps_location()
            addr = get_address(lat, lon)
            speak(f"You are near {addr}")
        elif "nearest clinic" in cmd:
            lat, lon = get_gps_location()
            place = search_place_nearby(lat, lon, "clinic")
            speak(f"Nearest clinic is at {place}" if place else "Clinic not found")
        else:
            speak("Command not recognized. Try again.")


if __name__ == "__main__":
    main()
