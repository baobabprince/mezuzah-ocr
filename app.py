import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from data_preprocessing import MezuzahDataPreprocessor
from letter_validator import HebrewLetterValidator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class MezuzahOCRApp:
    def __init__(self):
        self.preprocessor = MezuzahDataPreprocessor()
        self.model = self.load_model()
    
    def load_model(self):
        """
        Load pre-trained OCR model for Hebrew letters
        """
        try:
            # Placeholder for model loading
            # In a real implementation, you'd load a trained TensorFlow/Keras model
            model = tf.keras.models.load_model('hebrew_ocr_model.h5')
            return model
        except Exception as e:
            print(f"Model loading error: {e}")
            return None
    
    def predict_letters(self, letter_images):
        """
        Predict Hebrew letters from segmented images
        """
        if not self.model:
            raise ValueError("No OCR model loaded")
        
        # Prepare images for prediction
        processed_letters = np.array([
            cv2.resize(img, (224, 224)) for img in letter_images
        ])
        
        # Predict letters
        predictions = self.model.predict(processed_letters)
        return predictions

@app.route('/validate_mezuzah', methods=['POST'])
def validate_mezuzah():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Process mezuzah
    ocr_app = MezuzahOCRApp()
    preprocessor = MezuzahDataPreprocessor()
    
    try:
        # Preprocess image
        processed_image = preprocessor.preprocess_image(filepath)
        
        # Segment letters
        letter_images = preprocessor.segment_letters(processed_image)
        
        # Predict letters (placeholder - would use trained model)
        predicted_letters = [
            ('א', img) for img in letter_images  # Dummy prediction
        ]
        
        # Validate scroll
        validation_result = HebrewLetterValidator.validate_scroll(
            [(img, char) for char, img in predicted_letters]
        )
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(validation_result)
    
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
