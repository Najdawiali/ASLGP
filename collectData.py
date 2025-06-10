import os
import cv2
import numpy as np
import pickle
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    max_num_hands=2
)

DATA_DIR = './sequence_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

SEQUENCE_LENGTH = 30
NUM_CLASSES = 8
SEQUENCES_PER_CLASS = 20
FEATURE_LENGTH = 84


def extract_hand_features(frame):
    data_aux = np.zeros(FEATURE_LENGTH, dtype=np.float32)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    H, W, _ = frame.shape
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= 2:
                break

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            x_ = []
            y_ = []
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            if x_ and y_:
                min_x, min_y = min(x_), min(y_)

                base_idx = hand_idx * 42
                for i, landmark in enumerate(hand_landmarks.landmark):
                    data_aux[base_idx + i * 2] = landmark.x - min_x
                    data_aux[base_idx + i * 2 + 1] = landmark.y - min_y

    return data_aux, frame


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

all_sequences = []
all_labels = []

for class_idx in range(94,101):
    class_dir = os.path.join(DATA_DIR, str(class_idx))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {class_idx}')
    print(f'Press "Q" when ready to start collecting {SEQUENCES_PER_CLASS} sequences')

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        _, processed_frame = extract_hand_features(frame)

        cv2.putText(processed_frame, f'Press "Q" to collect sequences for class {class_idx}',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Sequence Collection', processed_frame)

        if cv2.waitKey(25) == ord('q'):
            break

    for seq_idx in range(SEQUENCES_PER_CLASS):
        print(f'Recording sequence {seq_idx + 1}/{SEQUENCES_PER_CLASS} for class {class_idx}')

        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            if not ret:
                continue

            _, processed_frame = extract_hand_features(frame)
            cv2.putText(processed_frame, f' Starting in {countdown}...',
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Sequence Collection', processed_frame)
            cv2.waitKey(1000)  # Wait 1 second

        current_sequence = []
        sequence_frames = []

        for frame_idx in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                continue


            features, processed_frame = extract_hand_features(frame)
            current_sequence.append(features)

            sequence_frames.append(frame.copy())

            progress = int((frame_idx + 1) / SEQUENCE_LENGTH * 100)
            cv2.putText(processed_frame, f'Recording: {progress}% (Frame {frame_idx + 1}/{SEQUENCE_LENGTH})',
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Sequence Collection', processed_frame)

            cv2.waitKey(30)

        current_sequence_array = np.array(current_sequence)
        print(f"Sequence shape: {current_sequence_array.shape}")

        sequence_path = os.path.join(class_dir, f'sequence_{seq_idx}.pickle')
        with open(sequence_path, 'wb') as f:
            pickle.dump({
                'features': current_sequence_array,
                'class': class_idx
            }, f)

        if len(sequence_frames) > 0:
            montage_dir = os.path.join(class_dir, 'montages')
            if not os.path.exists(montage_dir):
                os.makedirs(montage_dir)

            key_frames = [sequence_frames[0], sequence_frames[len(sequence_frames) // 2], sequence_frames[-1]]
            montage = np.hstack(key_frames)
            cv2.imwrite(os.path.join(montage_dir, f'sequence_{seq_idx}_montage.jpg'), montage)

        all_sequences.append(current_sequence_array)
        all_labels.append(class_idx)

        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, f'Sequence {seq_idx + 1} complete! Next in 2 seconds...',
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Sequence Collection', frame)
            cv2.waitKey(2000)  # Wait 2 seconds between sequences

    print(f'Finished collecting {SEQUENCES_PER_CLASS} sequences for class {class_idx}')

print(f'Saving complete dataset with {len(all_sequences)} sequences')
all_sequences_array = np.array(all_sequences)
all_labels_array = np.array(all_labels)
print(f"Final dataset shape: {all_sequences_array.shape}")

with open('sequence_data.pickle', 'wb') as f:
    pickle.dump({
        'data': all_sequences_array,
        'labels': all_labels_array
    }, f)

print('Done')
cap.release()
cv2.destroyAllWindows()
