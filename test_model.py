import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from data_preprocessing import MezuzahDataPreprocessor

class STAMPredictor:
    def __init__(self, model_path='stam_model.h5', mapping_path='label_mapping.npy'):
        """Initialize the STAM predictor"""
        self.model = tf.keras.models.load_model(model_path)
        self.idx_to_label = np.load(mapping_path, allow_pickle=True).item()
        self.preprocessor = MezuzahDataPreprocessor()
    
    def preprocess_image(self, image_path):
        """Preprocess a single image for prediction"""
        # Use the same preprocessing as training
        img = self.preprocessor.preprocess_image(image_path)
        return img.reshape(1, 128, 128, 1)
    
    def predict_letter(self, image_path):
        """Predict the Hebrew letter in an image"""
        # Preprocess image
        img = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        
        # Get predicted letter
        predicted_letter = self.idx_to_label[predicted_idx]
        
        return predicted_letter, confidence
    
    def visualize_prediction(self, image_path):
        """Visualize the image and prediction"""
        # Load and predict
        img = Image.open(image_path).convert('L')
        letter, confidence = self.predict_letter(image_path)
        
        # Create figure
        plt.figure(figsize=(6, 4))
        plt.imshow(img, cmap='gray')
        plt.title(f'Predicted: {letter} (Confidence: {confidence:.2%})')
        plt.axis('off')
        plt.show()

def test_on_directory(predictor, test_dir):
    """Test the model on a directory of images"""
    results = []
    
    for filename in os.listdir(test_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, filename)
            try:
                letter, confidence = predictor.predict_letter(image_path)
                results.append({
                    'filename': filename,
                    'predicted': letter,
                    'confidence': confidence
                })
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return results

if __name__ == "__main__":
    try:
        # Initialize predictor
        predictor = STAMPredictor()
        
        # Test on a single image
        test_image = "path_to_test_image.png"  # Replace with your test image path
        if os.path.exists(test_image):
            print("\nTesting single image:")
            letter, confidence = predictor.predict_letter(test_image)
            print(f"Predicted letter: {letter}")
            print(f"Confidence: {confidence:.2%}")
            predictor.visualize_prediction(test_image)
        
        # Test on a directory
        test_dir = "test_images"  # Replace with your test directory
        if os.path.exists(test_dir):
            print("\nTesting directory:")
            results = test_on_directory(predictor, test_dir)
            
            # Print results
            print("\nResults:")
            for result in results:
                print(f"File: {result['filename']}")
                print(f"Predicted: {result['predicted']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print("-" * 30)
    
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        exit(1)
