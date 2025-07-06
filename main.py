# AudioVision All-in-One Script (Modular Functions, Single File)

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
tts = pyttsx3.init()
tts.setProperty("rate", 125)


def speak(text):
    print("🗣️", text)
    tts.say(text)
    tts.runAndWait()


# === VOSK VOICE ===
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
            command = ""
            start = sd.time()
            while sd.time() - start < 7:
                data = audio_q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    command = result.get("text", "").lower()
                    if command:
                        print("Command:", command)
                        return command
            speak("I didn't catch that. Please try again.")


# === GPS ===
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


# === GOOGLE MAPS ===
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


# === OBJECT & STAIRS DETECTION ===
def object_detection_thread():
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    stairs_net = cv2.dnn.readNet(
        "stairs-yolov3-tiny_6500.weights", "stairs-yolov3-tiny.cfg"
    )
    CLASSES = open("coco.names").read().strip().split("\n")
    STAIRS_CLASSES = open("stairs.names").read().strip().split("\n")
    ALLOWED_CLASSES = {"person", "car", "bus", "chair", "sofa"}
    FOCAL_LENGTH = (325 * 130) / 55
    cap = cv2.VideoCapture(0)
    last_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (256, 256), swapRB=True, crop=False
        )

        # General Object
        net.setInput(blob)
        detections = net.forward(net.getUnconnectedOutLayersNames())
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and CLASSES[class_id] in ALLOWED_CLASSES:
                    box = obj[0:4] * np.array([width, height, width, height])
                    (_, _, box_w, _) = box.astype("int")
                    steps = int((55 * FOCAL_LENGTH) / box_w / 50)
                    if steps < 15 and time.time() - last_time > 5:
                        speak(f"{CLASSES[class_id]} ahead in {steps} steps")
                        last_time = time.time()

        # Stairs
        stairs_blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (256, 256), swapRB=True, crop=False
        )
        stairs_net.setInput(stairs_blob)
        stairs_output = stairs_net.forward(stairs_net.getUnconnectedOutLayersNames())
        for detection in stairs_output:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    box = obj[0:4] * np.array([width, height, width, height])
                    (_, _, box_w, _) = box.astype("int")
                    steps = int((55 * FOCAL_LENGTH) / box_w / 50)
                    if steps < 15 and time.time() - last_time > 5:
                        speak(f"{STAIRS_CLASSES[class_id]} ahead in {steps} steps")
                        last_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


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
