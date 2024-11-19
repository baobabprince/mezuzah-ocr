import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

class STAMModelTrainer:
    def __init__(self, dataset_dir: str = "stam_dataset"):
        self.dataset_dir = dataset_dir
        self.image_size = (128, 128)
        self.batch_size = 32
        self.epochs = 50
        
    def load_dataset(self):
        """Load and preprocess the dataset"""
        images = []
        labels = []
        label_to_idx = {}
        
        # Get all letter directories
        letter_dirs = [d for d in os.listdir(self.dataset_dir) 
                      if os.path.isdir(os.path.join(self.dataset_dir, d))]
        
        # Create label mapping
        for idx, letter in enumerate(sorted(letter_dirs)):
            label_to_idx[letter] = idx
        
        # Load images and labels
        for letter in letter_dirs:
            letter_path = os.path.join(self.dataset_dir, letter)
            for img_name in os.listdir(letter_path):
                img_path = os.path.join(letter_path, img_name)
                try:
                    # Load and convert to grayscale
                    img = Image.open(img_path).convert('L')
                    img = np.array(img) / 255.0  # Normalize
                    images.append(img)
                    labels.append(label_to_idx[letter])
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        return np.array(images), np.array(labels), label_to_idx
    
    def create_model(self, num_classes):
        """Create CNN model architecture"""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def train(self):
        """Train the model"""
        # Load dataset
        print("Loading dataset...")
        X, y, label_to_idx = self.load_dataset()
        idx_to_label = {v: k for k, v in label_to_idx.items()}
        
        # Reshape images for CNN
        X = X.reshape(-1, 128, 128, 1)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and compile model
        print("Creating model...")
        model = self.create_model(len(label_to_idx))
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        print("Training model...")
        history = model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_test, y_test)
        )
        
        # Evaluate model
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"\nTest accuracy: {test_acc:.4f}")
        
        # Save model and label mapping
        model.save('stam_model.h5')
        np.save('label_mapping.npy', idx_to_label)
        
        # Plot training history
        self.plot_training_history(history)
        
        return model, idx_to_label
    
    def plot_training_history(self, history):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

if __name__ == "__main__":
    try:
        trainer = STAMModelTrainer()
        model, label_mapping = trainer.train()
        print("\nTraining complete! Model saved as 'stam_model.h5'")
        print("Label mapping saved as 'label_mapping.npy'")
        print("Training history plot saved as 'training_history.png'")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        exit(1)
