import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import random
from typing import Tuple, List
import cv2

class STAMDatasetGenerator:
    def __init__(self, output_dir: str = "dataset"):
        """
        Initialize the STAM dataset generator
        
        Args:
            output_dir (str): Directory to save generated images
        """
        self.output_dir = output_dir
        self.image_size = (128, 128)
        self.background_color = (255, 255, 255)  # White
        self.text_color = (0, 0, 0)  # Black
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        for char in self.get_hebrew_letters():
            os.makedirs(os.path.join(output_dir, char), exist_ok=True)
    
    @staticmethod
    def get_hebrew_letters() -> List[str]:
        """Get list of Hebrew letters used in STAM"""
        return [
            'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י',
            'כ', 'ך', 'ל', 'מ', 'ם', 'נ', 'ן', 'ס', 'ע', 'פ',
            'ף', 'צ', 'ץ', 'ק', 'ר', 'ש', 'ת'
        ]
    
    def apply_random_noise(self, image: Image.Image) -> Image.Image:
        """Add random noise to image"""
        img_array = np.array(image)
        noise = np.random.normal(0, 25, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    def apply_random_blur(self, image: Image.Image) -> Image.Image:
        """Apply random Gaussian blur"""
        radius = random.uniform(0, 1.5)
        return image.filter(ImageFilter.GaussianBlur(radius))
    
    def apply_random_rotation(self, image: Image.Image) -> Image.Image:
        """Apply random rotation"""
        angle = random.uniform(-5, 5)
        return image.rotate(angle, expand=True)
    
    def apply_random_perspective(self, image: Image.Image) -> Image.Image:
        """Apply random perspective transformation"""
        width, height = image.size
        
        # Define perspective coefficients
        coeffs = [
            random.uniform(-0.1, 0.1) for _ in range(8)
        ]
        
        # Apply perspective transform
        return image.transform(
            (width, height),
            Image.PERSPECTIVE,
            coeffs,
            Image.BICUBIC
        )
    
    def apply_random_quality(self, image: Image.Image) -> Image.Image:
        """Randomly reduce image quality"""
        # Random JPEG compression
        quality = random.randint(60, 95)
        temp_path = f"temp_{random.randint(0, 999999)}.jpg"
        try:
            image.save(temp_path, "JPEG", quality=quality)
            image = Image.open(temp_path)
            return image
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
    
    def generate_letter_image(
        self, 
        letter: str, 
        font_path: str,
        font_size: int = 80
    ) -> Image.Image:
        """Generate base image with letter"""
        # Create image with padding
        padding = 20
        size = (font_size + padding * 2, font_size + padding * 2)
        image = Image.new('L', size, 255)
        draw = ImageDraw.Draw(image)
        
        # Load font and draw letter
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            raise ValueError(f"Could not load font at {font_path}")
        
        # Center the letter
        bbox = draw.textbbox((0, 0), letter, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        
        draw.text((x, y), letter, font=font, fill=0)
        return image
    
    def generate_dataset(
        self,
        font_path: str,
        samples_per_letter: int = 1000
    ):
        """
        Generate dataset with variations for each letter
        
        Args:
            font_path (str): Path to STAM font file
            samples_per_letter (int): Number of samples to generate per letter
        """
        letters = self.get_hebrew_letters()
        
        for letter in letters:
            print(f"Generating samples for letter {letter}")
            letter_dir = os.path.join(self.output_dir, letter)
            
            for i in range(samples_per_letter):
                # Generate base image
                image = self.generate_letter_image(letter, font_path)
                
                # Apply random transformations
                if random.random() > 0.3:
                    image = self.apply_random_rotation(image)
                if random.random() > 0.3:
                    image = self.apply_random_perspective(image)
                if random.random() > 0.3:
                    image = self.apply_random_blur(image)
                if random.random() > 0.3:
                    image = self.apply_random_noise(image)
                if random.random() > 0.3:
                    image = self.apply_random_quality(image)
                
                # Resize to final size
                image = image.resize(self.image_size, Image.LANCZOS)
                
                # Save image
                image.save(os.path.join(letter_dir, f"{letter}_{i}.png"))
        
        print("Dataset generation complete!")

if __name__ == "__main__":
    try:
        # Example usage
        generator = STAMDatasetGenerator("stam_dataset")
        
        # Check if font exists
        font_path = "C:/Windows/Fonts/STAM1.ttf"
        if not os.path.exists(font_path):
            print(f"Error: Font file not found at {font_path}")
            print("Please ensure you have installed a STAM font and provided the correct path.")
            exit(1)
            
        # Generate dataset
        print(f"Starting dataset generation using font: {font_path}")
        generator.generate_dataset(
            font_path=font_path,
            samples_per_letter=1000
        )
    except Exception as e:
        print(f"Error during dataset generation: {str(e)}")
        exit(1)
