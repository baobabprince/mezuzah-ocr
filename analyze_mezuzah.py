import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from data_preprocessing import MezuzahDataPreprocessor
from test_model import STAMPredictor

class MezuzahAnalyzer:
    def __init__(self, model_path='stam_model.h5', mapping_path='label_mapping.npy'):
        """Initialize the Mezuzah analyzer"""
        self.predictor = STAMPredictor(model_path, mapping_path)
        self.preprocessor = MezuzahDataPreprocessor()
    
    def preprocess_full_image(self, image_path):
        """Preprocess the full mezuzah image"""
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Show original image
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        
        # Apply preprocessing
        # Enhance contrast
        img = cv2.equalizeHist(img)
        
        # Denoise
        img = cv2.fastNlMeansDenoising(img)
        
        # Binarization with Otsu's method
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Show preprocessed image
        plt.subplot(122)
        plt.imshow(img, cmap='gray')
        plt.title('Preprocessed Image')
        plt.show()
        
        return img

    def segment_lines(self, image):
        """Segment the image into lines of text"""
        # Calculate horizontal projection
        h_proj = np.sum(image == 0, axis=1)
        
        # Visualize horizontal projection
        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        plt.plot(h_proj, range(len(h_proj)))
        plt.gca().invert_yaxis()
        plt.title('Horizontal Projection')
        
        # Find line boundaries
        lines = []
        in_line = False
        start = 0
        
        for i, count in enumerate(h_proj):
            if count > 0 and not in_line:
                start = i
                in_line = True
            elif count == 0 and in_line:
                lines.append((start, i))
                in_line = False
        
        if in_line:
            lines.append((start, len(h_proj)))
        
        # Visualize line segmentation
        plt.subplot(122)
        plt.imshow(image, cmap='gray')
        for start, end in lines:
            plt.axhline(y=start, color='r', linestyle='-', alpha=0.3)
            plt.axhline(y=end, color='r', linestyle='-', alpha=0.3)
        plt.title('Detected Lines')
        plt.show()
        
        return lines

    def segment_letters(self, image, line_coords):
        """Segment a line into individual letters"""
        start_y, end_y = line_coords
        line_image = image[start_y:end_y]
        
        # Calculate vertical projection
        v_proj = np.sum(line_image == 0, axis=0)
        
        # Find letter boundaries
        letters = []
        in_letter = False
        start = 0
        min_width = 10  # Minimum width to consider as a letter
        
        for i, count in enumerate(v_proj):
            if count > 0 and not in_letter:
                start = i
                in_letter = True
            elif count == 0 and in_letter:
                if i - start >= min_width:
                    letters.append((start, i))
                in_letter = False
        
        if in_letter and len(line_image) - start >= min_width:
            letters.append((start, len(v_proj)))
        
        return letters

    def extract_letter_image(self, image, line_coords, letter_coords):
        """Extract a single letter from the image"""
        start_y, end_y = line_coords
        start_x, end_x = letter_coords
        
        # Extract letter with padding
        padding = 5
        start_y = max(0, start_y - padding)
        end_y = min(image.shape[0], end_y + padding)
        start_x = max(0, start_x - padding)
        end_x = min(image.shape[1], end_x + padding)
        
        letter_img = image[start_y:end_y, start_x:end_x]
        
        # Add padding if necessary to make it square
        height, width = letter_img.shape
        max_dim = max(height, width)
        
        # Create a square white background
        square_img = np.full((max_dim, max_dim), 255, dtype=np.uint8)
        
        # Center the letter in the square
        y_offset = (max_dim - height) // 2
        x_offset = (max_dim - width) // 2
        square_img[y_offset:y_offset+height, x_offset:x_offset+width] = letter_img
        
        # Resize to model input size
        letter_img = cv2.resize(square_img, (128, 128), interpolation=cv2.INTER_AREA)
        
        return letter_img

    def analyze_image(self, image_path, visualize=True):
        """Analyze a full mezuzah image"""
        # Preprocess image
        print("Preprocessing image...")
        img = self.preprocess_full_image(image_path)
        
        # Segment into lines
        print("Segmenting lines...")
        lines = self.segment_lines(img)
        
        # Process each line
        results = []
        
        for line_idx, line_coords in enumerate(lines):
            print(f"Processing line {line_idx + 1}...")
            # Segment line into letters
            letters = self.segment_letters(img, line_coords)
            
            line_results = []
            for letter_coords in letters:
                # Extract letter
                letter_img = self.extract_letter_image(img, line_coords, letter_coords)
                
                # Ensure correct shape for model
                letter_img_reshaped = letter_img.reshape(1, 128, 128, 1)
                
                # Predict letter
                try:
                    predictions = self.predictor.model.predict(letter_img_reshaped)
                    predicted_idx = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_idx]
                    letter = self.predictor.idx_to_label[predicted_idx]
                    
                    line_results.append({
                        'letter': letter,
                        'confidence': confidence,
                        'coords': (line_coords, letter_coords)
                    })
                except Exception as e:
                    print(f"Error predicting letter: {e}")
            
            results.append(line_results)
        
        if visualize:
            self.visualize_results(img, results)
        
        return results

    def visualize_results(self, image, results):
        """Visualize the analysis results"""
        # Convert to RGB for visualization
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Draw boxes and predictions
        for line_results in results:
            for result in line_results:
                (line_y1, line_y2), (letter_x1, letter_x2) = result['coords']
                
                # Draw rectangle
                cv2.rectangle(vis_img, 
                            (letter_x1, line_y1),
                            (letter_x2, line_y2),
                            (0, 255, 0), 2)
                
                # Add text
                text = f"{result['letter']} ({result['confidence']:.2f})"
                cv2.putText(vis_img, text,
                          (letter_x1, line_y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (255, 0, 0), 1)
        
        # Display
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Mezuzah Analysis Results')
        plt.show()

if __name__ == "__main__":
    try:
        # Initialize analyzer
        analyzer = MezuzahAnalyzer()
        
        # Analyze image
        image_path = input("Enter the path to your mezuzah image: ")
        if not os.path.exists(image_path):
            raise ValueError(f"Image not found: {image_path}")
        
        print("\nAnalyzing mezuzah image...")
        results = analyzer.analyze_image(image_path)
        
        # Print results
        print("\nAnalysis Results:")
        for line_idx, line_results in enumerate(results):
            print(f"\nLine {line_idx + 1}:")
            line_text = ""
            for result in line_results:
                line_text += result['letter']
            print(f"Text: {line_text}")
            print("Letter Details:")
            for result in line_results:
                print(f"  {result['letter']}: {result['confidence']:.2%} confidence")
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        exit(1)
