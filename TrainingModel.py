import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


class HandLandmarkAugmenter:
    
    def __init__(self, 
                 translation_range=0.1,
                 scale_range=0.2,
                 rotation_range=15,
                 noise_std=0.01):
      
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.noise_std = noise_std
        
    
    def normalize_landmarks(self, landmarks):
        
        normalized = landmarks.copy()
        timesteps, n_features = landmarks.shape
        
        for t in range(timesteps):
            frame_landmarks = landmarks[t].reshape(-1, 2)
            
            if len(frame_landmarks) > 0:
                min_x, min_y = np.min(frame_landmarks, axis=0)
                max_x, max_y = np.max(frame_landmarks, axis=0)
                
                range_x = max_x - min_x
                range_y = max_y - min_y
                
                range_x = max(range_x, 0.001)
                range_y = max(range_y, 0.001)
                
                frame_landmarks[:, 0] = (frame_landmarks[:, 0] - min_x) / range_x
                frame_landmarks[:, 1] = (frame_landmarks[:, 1] - min_y) / range_y
                
                
                normalized[t] = frame_landmarks.flatten()
        
        return normalized
    
    def apply_translation(self, landmarks):
        augmented = landmarks.copy()
        timesteps, n_features = landmarks.shape
        
        tx = np.random.uniform(-self.translation_range, self.translation_range)
        ty = np.random.uniform(-self.translation_range, self.translation_range)
        
        for t in range(timesteps):
            frame_landmarks = augmented[t].reshape(-1, 2)
            frame_landmarks[:, 0] += tx
            frame_landmarks[:, 1] += ty
            augmented[t] = frame_landmarks.flatten()
        
        return augmented
    
    def apply_scaling(self, landmarks):
        augmented = landmarks.copy()
        timesteps, n_features = landmarks.shape
        
        scale_x = np.random.uniform(1.0 - self.scale_range, 1.0 + self.scale_range)
        scale_y = np.random.uniform(1.0 - self.scale_range, 1.0 + self.scale_range)
        
        for t in range(timesteps):
            frame_landmarks = augmented[t].reshape(-1, 2)
            
            centroid_x = np.mean(frame_landmarks[:, 0])
            centroid_y = np.mean(frame_landmarks[:, 1])
            
            frame_landmarks[:, 0] = (frame_landmarks[:, 0] - centroid_x) * scale_x + centroid_x
            frame_landmarks[:, 1] = (frame_landmarks[:, 1] - centroid_y) * scale_y + centroid_y
            augmented[t] = frame_landmarks.flatten()
        
        return augmented
    
    def apply_rotation(self, landmarks):
        augmented = landmarks.copy()
        timesteps, n_features = landmarks.shape
        
        angle = np.random.uniform(-self.rotation_range, self.rotation_range) * np.pi / 180.0
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        for t in range(timesteps):
            frame_landmarks = augmented[t].reshape(-1, 2)
            
            centroid_x = np.mean(frame_landmarks[:, 0])
            centroid_y = np.mean(frame_landmarks[:, 1])
            
            for i in range(len(frame_landmarks)):
                x = frame_landmarks[i, 0] - centroid_x
                y = frame_landmarks[i, 1] - centroid_y
                
                frame_landmarks[i, 0] = x * cos_angle - y * sin_angle + centroid_x
                frame_landmarks[i, 1] = x * sin_angle + y * cos_angle + centroid_y
            
            augmented[t] = frame_landmarks.flatten()
        
        return augmented
    
    def apply_noise(self, landmarks):
        
        noise = np.random.normal(0, self.noise_std, landmarks.shape)
        return landmarks + noise
    
    
    def temporal_dropout(self, landmarks, dropout_rate=0.1):
        
        augmented = landmarks.copy()
        timesteps = landmarks.shape[0]
        n_frames_to_drop = int(timesteps * dropout_rate)
        frames_to_drop = np.random.choice(timesteps, n_frames_to_drop, replace=False)
        
        for frame_idx in frames_to_drop:
            augmented[frame_idx] = 0 
        
        return augmented
    
    def augment_sequence(self, landmarks, augmentation_probability=0.7):
        
        augmented = landmarks.copy()
        
        augmented = self.normalize_landmarks(augmented)
        
        if np.random.random() < augmentation_probability:
            augmented = self.apply_translation(augmented)
        
        if np.random.random() < augmentation_probability:
            augmented = self.apply_scaling(augmented)
        
        if np.random.random() < augmentation_probability:
            augmented = self.apply_rotation(augmented)
        
        if np.random.random() < augmentation_probability:
            augmented = self.apply_noise(augmented)
        
        if np.random.random() < 0.3:  
            augmented = self.temporal_dropout(augmented)
        
        return augmented

def augment_dataset(X, y, augmenter, augmentation_factor=2):
    
    print(f"Augmenting dataset with factor {augmentation_factor}")
    
    X_augmented = [X]  
    y_augmented = [y]
    
    for aug_iter in range(augmentation_factor):
        print(f"Generating augmentation batch {aug_iter + 1}/{augmentation_factor}...")
        
        X_aug_batch = []
        for i, sequence in enumerate(X):
            if i % 100 == 0:
                print(f"  Processing sample {i}/{len(X)}")
            
            augmented_sequence = augmenter.augment_sequence(sequence)
            X_aug_batch.append(augmented_sequence)
        
        X_augmented.append(np.array(X_aug_batch))
        y_augmented.append(y)  
    
    X_final = np.concatenate(X_augmented, axis=0)
    y_final = np.concatenate(y_augmented, axis=0)
    
    print(f"Dataset size: {len(X)} -> {len(X_final)} samples")
    return X_final, y_final

with open('sequence_datafor101.pickle', 'rb') as f:
    data_dict = pickle.load(f)

sequence_data = np.array(data_dict['data'])
sequence_labels = np.array(data_dict['labels'])

print(f"Original dataset shape: {sequence_data.shape}")
print(f"Number of classes: {len(np.unique(sequence_labels))}")

categorical_labels = to_categorical(sequence_labels, num_classes=101)

n_samples, timesteps, n_features = sequence_data.shape

augmenter = HandLandmarkAugmenter(
    translation_range=0.15,   
    scale_range=0.25,            
    rotation_range=20,          
    noise_std=0.005,             
)

# Apply data augmentation
augmented_X, augmented_y = augment_dataset(
    sequence_data, 
    categorical_labels, 
    augmenter, 
    augmentation_factor=3  
)

print(f"Augmented dataset shape: {augmented_X.shape}")

# Split the augmented data
X_train, X_test, y_train, y_test = train_test_split(
    augmented_X, augmented_y, test_size=0.2,
    shuffle=True, stratify=np.argmax(augmented_y, axis=1), random_state=42
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

X_train_reshaped = X_train.reshape(-1, n_features)
X_test_reshaped = X_test.reshape(-1, n_features)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)

X_train_scaled = X_train_scaled.reshape(-1, timesteps, n_features)
X_test_scaled = X_test_scaled.reshape(-1, timesteps, n_features)

model = Sequential([
    GRU(128, return_sequences=True, input_shape=(timesteps, n_features)),
    Dropout(0.3),
    GRU(64, return_sequences=True),
    Dropout(0.3),
    GRU(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(101, activation='softmax')
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,  
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    'best_gru_model_augmented.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

model.summary()

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=100,  
    batch_size=64,  
    callbacks=[early_stopping, model_checkpoint, lr_reduction],
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

model.save('gru_model_augmented.h5')  
with open('gru_preprocessing_augmented.pickle', 'wb') as f:
    pickle.dump({
        'scaler': scaler,
        'label_encoder': sequence_labels,
        'n_classes': 101,
        'timesteps': timesteps,
        'n_features': n_features
    }, f)

y_pred = model.predict(X_test_scaled, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

num_classes = y_test.shape[1]

class_names = [str(i) for i in range(num_classes)]

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, 
                          target_names=class_names))


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
