import cv2
import numpy as np
import tensorflow as tf
from typing import List, Tuple

class MezuzahDataPreprocessor:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess mezuzah scroll image for OCR
        
        Args:
            image_path (str): Path to the mezuzah scroll image
        
        Returns:
            np.ndarray: Preprocessed image
        """
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Enhance contrast
        img = cv2.equalizeHist(img)
        
        # Denoise
        img = cv2.fastNlMeansDenoising(img)
        
        # Binarization
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Resize
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        
        # Normalize
        img = img / 255.0
        
        return img
    
    def segment_letters(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Segment individual letters from the mezuzah scroll
        
        Args:
            image (np.ndarray): Preprocessed image
        
        Returns:
            List[np.ndarray]: List of segmented letter images
        """
        # Contour detection
        contours, _ = cv2.findContours(
            (image * 255).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and extract letters
        letters = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out noise and small contours
            if w > 10 and h > 10:
                letter_img = image[y:y+h, x:x+w]
                letters.append(cv2.resize(letter_img, self.image_size))
        
        return letters
