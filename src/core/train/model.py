import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..')
sys.path.append(root_dir)

from src.utils.constants import LENGHT_KEYPOINTS

from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Conv1D, MaxPooling1D, GlobalAveragePooling1D # type: ignore
from keras.regularizers import l2 # type: ignore
from keras.optimizers import Adam # type: ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore


def get_model(max_length_frames, output_length: int):
    """
    Enhanced model architecture for sign language recognition
    Uses bidirectional LSTMs with attention mechanism and proper regularization
    """
    model = Sequential()

    model.add(Conv1D(64, 3, activation='relu', input_shape=(max_length_frames, LENGHT_KEYPOINTS)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
    model.add(BatchNormalization())
    
    model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling1D())

    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    
    model.add(Dense(output_length, activation='softmax'))
    
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    return model


def get_lightweight_model(max_length_frames, output_length: int):
    """
    Lightweight model for faster inference with good performance
    """
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=(max_length_frames, LENGHT_KEYPOINTS)))
    model.add(Dropout(0.3))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(output_length, activation='softmax'))
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_small_dataset_model(max_length_frames, output_length: int):
    """
    Model optimized for very small datasets (< 100 samples)
    Uses minimal complexity to avoid overfitting
    """
    model = Sequential()

    model.add(Conv1D(32, 3, activation='relu', input_shape=(max_length_frames, LENGHT_KEYPOINTS)))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))
    
    model.add(LSTM(64, return_sequences=False, dropout=0.2))
    model.add(Dropout(0.3))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(output_length, activation='softmax'))

    optimizer = Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks():
    """
    Get training callbacks for better training control
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1
    )
    
    return [early_stopping, reduce_lr]


def get_data_augmentation_callbacks():
    """
    Custom data augmentation for keypoints data
    """
    def noise_augmentation(x, noise_level=0.01):
        """Add small amount of noise to keypoints"""
        import numpy as np
        noise = np.random.normal(0, noise_level, x.shape)
        return x + noise
    
    def time_shift_augmentation(x, shift_range=2):
        """Shift sequence in time"""
        import numpy as np
        shift = np.random.randint(-shift_range, shift_range + 1)
        if shift > 0:
            return np.concatenate([x[shift:], np.zeros((shift, x.shape[1]))])
        elif shift < 0:
            return np.concatenate([np.zeros((-shift, x.shape[1])), x[:shift]])
        return x
    
    return noise_augmentation, time_shift_augmentation