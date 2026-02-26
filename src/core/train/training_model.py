from src.core.train.model import get_model, get_small_dataset_model, get_lightweight_model, get_callbacks
from src.utils.utils import get_word_ids, get_sequences_and_labels
from src.utils.constants import *

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def augment_sequences(sequences, labels, augmentation_factor=3):
    """
    Apply data augmentation to increase dataset size
    """
    augmented_sequences = []
    augmented_labels = []

    augmented_sequences.extend(sequences)
    augmented_labels.extend(labels)

    for i in range(augmentation_factor):
        for seq, label in zip(sequences, labels):
            noise = np.random.normal(0, 0.01, seq.shape)
            noisy_seq = seq + noise

            shift = np.random.randint(-2, 3)
            if shift > 0:
                shifted_seq = np.concatenate([noisy_seq[shift:], np.zeros((shift, noisy_seq.shape[1]))])
            elif shift < 0:
                shifted_seq = np.concatenate([np.zeros((-shift, noisy_seq.shape[1])), noisy_seq[:shift]])
            else:
                shifted_seq = noisy_seq
            
            augmented_sequences.append(shifted_seq)
            augmented_labels.append(label)
    
    return augmented_sequences, augmented_labels

def training_model(model_path, model_num: int, epochs=NUM_EPOCHS, use_lightweight=False):
    word_ids = get_word_ids(KEYPOINTS_PATH)
    
    sequences, labels = get_sequences_and_labels(word_ids, model_num)
    
    if not sequences:
        print("No valid sequences found for training!")
        return
    
    unique_labels = set(labels)
    num_classes = len(unique_labels)
    
    print(f"Training with {len(sequences)} sequences and {num_classes} classes")
    print(f"Classes: {unique_labels}")
    
    sequences = pad_sequences(sequences, maxlen=int(model_num), padding='pre', truncating='post', dtype='float32')
    
    X = np.array(sequences)
    y = to_categorical(labels, num_classes=num_classes).astype(int)

    if len(sequences) > 4:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=labels)
        validation_data = (X_val, y_val)
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
    else:
        X_train, y_train = X, y
        validation_data = None
        print("Warning: Not enough data for validation split. Using all data for training.")

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class weights: {class_weight_dict}")
    
    if use_lightweight:
        model = get_lightweight_model(int(model_num), num_classes)
        print("Using lightweight model")
    else:
        if len(sequences) < 50:
            print("Using small dataset model (optimized for few samples)")
            model = get_small_dataset_model(int(model_num), num_classes)

            print("Applying data augmentation...")
            sequences, labels = augment_sequences(sequences, labels, augmentation_factor=5)
            print(f"Dataset size after augmentation: {len(sequences)} samples")
        else:
            print("Using enhanced model")
            model = get_model(int(model_num), num_classes)

    callbacks = get_callbacks() if validation_data else []
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=min(32, len(X_train)), 
        validation_data=validation_data,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    if validation_data:
        evaluation_results = model.evaluate(X_val, y_val, verbose=0)
        val_loss = evaluation_results[0]
        val_accuracy = evaluation_results[1]
        print(f"\nFinal validation accuracy: {val_accuracy:.4f}")
        print(f"Final validation loss: {val_loss:.4f}")

        if len(evaluation_results) > 2:
            print(f"Final validation top-k accuracy: {evaluation_results[2]:.4f}")
    
    model.summary()
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    return history

if __name__ == "__main__":
    model_num = 18
    model_path = os.path.join(MODELS_FOLDER_PATH, f'actions_{model_num}.keras')
    training_model(model_path, model_num)

