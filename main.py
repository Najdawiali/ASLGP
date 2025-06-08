import pickle
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import base64
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os


def load_model_components():
    try:
        with open('./lstm_preprocessing_augmentedV2.pickle', 'rb') as f:
            preproc = pickle.load(f)

        interpreter = tf.lite.Interpreter(model_path="gesture_model.tflite")
        interpreter.allocate_tensors()

        return interpreter, preproc
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


model, preproc = load_model_components()
scaler = preproc['scaler']
timesteps = preproc['timesteps']
n_features = preproc['n_features']

LABELS = {0: 'السلام عليكم', 1: 'صباح الخير', 2: 'شكراً', 3: 'أنا', 4: 'أنتَ', 5: 'أنتِ', 6: 'هو', 7: 'هي',
          8: 'أنتم', 9: 'هم', 10: 'اسم', 11: 'كيف حالك؟', 12: 'الحمد لله', 13: 'سعيد', 14: 'حزين', 15: 'غاضب',
          16: 'جيد', 17: 'سيء', 18: 'تعبان', 19: 'مريض', 20: 'أرى', 21: 'أقول', 22: 'أتكلم', 23: 'أمشي', 24: 'ذهبت',
          25: 'جاء', 26: 'بيت', 27: 'أكل', 28: 'نام', 29: 'الجامعة', 30: 'اليوم', 31: 'غداً', 32: 'الأحد', 33: 'الثلاثاء',
          34: 'الخميس', 35: 'الجمعة', 36: 'أسبوع', 37: 'شهر', 38: 'سنة', 39: 'متى', 40: 'أعرف', 41: 'أفكر', 42: 'نسيت',
          43: 'أحب', 44: 'أريد', 45: 'يساعد', 46: 'ممنوع', 47: 'أوافق', 48: 'معاً', 49: 'مختلف', 50: 'أ', 51: 'ن', 52: 'س', 53: 'ع',
          54: 'ل', 55: 'ي', 56: 'م', 57: 'ح', 58: 'د', 59: 'ف', 60: 'ر', 61: 'اثنين', 62: 'اين', 63: 'كيف', 64: 'لماذا',
          65: 'سبب', 66: 'ساعة', 67: 'خطر', 68: 'حرب', 69: 'دينار', 70: 'الله', 71: 'جنة', 72: 'ان شاء الله', 73: 'الجو', 74: 'منسف',
          75: 'دائماً', 76: 'سيارة', 77: 'جديد', 78: 'قديم', 79: 'صح', 80: 'خطأ', 81: 'الاردن', 82: 'اعطيني', 83: 'مبروك',
          84: 'ربع ساعة', 85: 'مستشفى', 86: 'دكتور', 87: 'وعليكم السلام', 88: 'اربد', 89: 'لا استطيع', 90: 'ملف', 91: 'شهادة',
          92: 'لابتوب', 93: 'منّذ', 94: 'مَنْ', 95: 'اسبانيا', 96: 'نادي', 97: 'نعم', 98: 'لا', 99: 'صلاة الظهر', 100: 'عيد الاضحى',
          }

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.63,
    min_tracking_confidence=0.6,
    max_num_hands=2
)


class SignLanguageProcessor:

    def __init__(self, confidence_threshold=0.7, max_frames_without_hands=10):
        self.sequence_buffer = deque(maxlen=timesteps)
        self.current_prediction = None
        self.frames_since_hands = 0
        self.confidence_threshold = confidence_threshold
        self.max_frames_without_hands = max_frames_without_hands

    def extract_features(self, frame):
        features = np.zeros(n_features, dtype=np.float32)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return features, False

        for hand_idx, landmarks in enumerate(results.multi_hand_landmarks[:2]):
            coords = [(lm.x, lm.y) for lm in landmarks.landmark]
            if len(coords) != 21:
                continue

            coords = np.array(coords)
            min_x, min_y = np.min(coords, axis=0)
            max_x, max_y = np.max(coords, axis=0)
            range_x = max(max_x - min_x, 0.001)
            range_y = max(max_y - min_y, 0.001)

            coords[:, 0] = (coords[:, 0] - min_x) / range_x
            coords[:, 1] = (coords[:, 1] - min_y) / range_y

            base_idx = hand_idx * 42
            for i, (x, y) in enumerate(coords):
                if base_idx + i * 2 + 1 < n_features:
                    features[base_idx + i * 2] = x
                    features[base_idx + i * 2 + 1] = y

        return features, True

    def process_frame(self, frame):
        features, hands_present = self.extract_features(frame)

        if hands_present:
            self.sequence_buffer.append(features)
            self.frames_since_hands = 0
        else:
            self.frames_since_hands += 1
            if self.frames_since_hands > self.max_frames_without_hands:
                self.current_prediction = None
                self.sequence_buffer.clear()

        confidence = 0

        if len(self.sequence_buffer) == timesteps:
            # Prepare and scale data
            sequence_data = np.array(list(self.sequence_buffer)).reshape(-1, n_features)
            sequence_scaled = scaler.transform(sequence_data).reshape(1, timesteps, n_features)

            input_details = model.get_input_details()
            output_details = model.get_output_details()

            input_data = sequence_scaled.astype(np.float32)

            model.set_tensor(input_details[0]['index'], input_data)

            model.invoke()

            output_data = model.get_tensor(output_details[0]['index'])[0]
            predicted_idx = np.argmax(output_data)
            confidence = output_data[predicted_idx]

            if confidence > self.confidence_threshold:
                self.current_prediction = LABELS[predicted_idx]
                self.sequence_buffer.clear()

        return {
            'prediction': self.current_prediction,
            'confidence': confidence,
            'hands_detected': hands_present,
            'buffer_level': len(self.sequence_buffer),
            'frames_since_hands': self.frames_since_hands
        }


app = FastAPI(title="Sign Language Recognition API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processors = {}


@app.get("/")
async def root():
    return {"message": "Sign Language Recognition API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    processors[websocket] = SignLanguageProcessor()

    try:
        while True:
            data = await websocket.receive_text()

            try:
                # Decode image
                image_data = base64.b64decode(data.split(",")[1])
                frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

                if frame is not None:
                    result = processors[websocket].process_frame(frame)

                    # Send comprehensive JSON response
                    await websocket.send_json({
                        "status": "prediction",
                        "prediction": result['prediction'],
                        "confidence": float(result['confidence']),
                        "hands_detected": result['hands_detected'],
                        "buffer_level": result['buffer_level'],
                        "frames_since_hands": result['frames_since_hands']
                    })

            except Exception as e:
                print(f"Frame processing error: {e}")
                await websocket.send_json({
                    "status": "error",
                    "prediction": None,
                    "confidence": 0.0,
                    "hands_detected": False,
                    "buffer_level": 0,
                    "error": str(e)
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        processors.pop(websocket, None)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)