import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import List, Tuple
import shutil
import glob

class STAMDatasetGenerator:
    def __init__(self, output_dir: str):
        """Initialize the dataset generator
        
        Args:
            output_dir (str): Directory to save the generated dataset
        """
        self.output_dir = output_dir
        self.hebrew_letters = [
            'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י',
            'כ', 'ך', 'ל', 'מ', 'ם', 'נ', 'ן', 'ס', 'ע', 'פ',
            'ף', 'צ', 'ץ', 'ק', 'ר', 'ש', 'ת'
        ]
        
        # Default font sizes for different fonts
        self.font_sizes = {
            'regular': 100,
            'small': 80,
            'large': 120
        }

    def load_fonts(self, font_dir: str) -> List[Tuple[ImageFont.FreeTypeFont, str]]:
        """Load only STAM TTF fonts from the specified directory
        
        Args:
            font_dir (str): Directory containing STAM fonts
            
        Returns:
            List[Tuple[ImageFont.FreeTypeFont, str]]: List of (font, font_name) tuples
        """
        fonts = []
        # Get all TTF files
        all_fonts = glob.glob(os.path.join(font_dir, "*.ttf"))
        
        # Filter for fonts starting with STAM (case insensitive)
        stam_fonts = []
        for font_path in all_fonts:
            font_name = os.path.basename(font_path).upper()
            if font_name.startswith('STAM'):
                stam_fonts.append(font_path)
        
        if not stam_fonts:
            raise ValueError(f"No STAM fonts found in {font_dir}. Please make sure you have fonts that start with 'STAM'")
            
        print(f"\nFound {len(stam_fonts)} STAM fonts:")
        for font_path in stam_fonts:
            print(f"  - {os.path.basename(font_path)}")
            
        for font_path in stam_fonts:
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            try:
                # Load font in different sizes
                for size_name, size in self.font_sizes.items():
                    font = ImageFont.truetype(font_path, size)
                    fonts.append((font, f"{font_name}_{size_name}"))
                    print(f"    Loaded {font_name} in {size_name} size ({size}px)")
            except Exception as e:
                print(f"Warning: Could not load font {font_path}: {str(e)}")
                
        if not fonts:
            raise ValueError("No usable STAM fonts were loaded. Please check if your STAM fonts are valid TTF files.")
            
        return fonts

    def generate_letter_image(self, letter: str, font: ImageFont.FreeTypeFont) -> Image.Image:
        """Generate base image with letter
        
        Args:
            letter (str): Hebrew letter to generate
            font (ImageFont.FreeTypeFont): Font to use for the letter
            
        Returns:
            Image.Image: Base image with the letter
        """
        # Create image with padding
        padding = 20
        size = (font.getsize(letter)[0] + padding * 2, font.getsize(letter)[1] + padding * 2)
        image = Image.new('L', size, 255)
        draw = ImageDraw.Draw(image)
        
        # Center the letter
        x = (size[0] - font.getsize(letter)[0]) // 2
        y = (size[1] - font.getsize(letter)[1]) // 2
        
        draw.text((x, y), letter, font=font, fill=0)
        return image

    def apply_random_distortions(self, image: Image.Image) -> Image.Image:
        """Apply subtle random distortions to the image
        
        Args:
            image (Image.Image): Image to distort
            
        Returns:
            Image.Image: Distorted image
        """
        # Apply random rotation (reduced from ±3° to ±1.5°)
        if random.random() > 0.5:
            angle = random.uniform(-1.5, 1.5)
            image = image.rotate(angle, expand=True)
        
        # Apply random perspective transformation (reduced from ±0.05 to ±0.02)
        if random.random() > 0.5:
            width, height = image.size
            coeffs = [random.uniform(-0.02, 0.02) for _ in range(8)]
            image = image.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
        
        # Apply random blur (reduced from 0-0.8 to 0-0.4)
        if random.random() > 0.5:
            radius = random.uniform(0, 0.4)
            image = image.filter(ImageFilter.GaussianBlur(radius))
        
        # Apply random noise (reduced from 0-10 to 0-5)
        if random.random() > 0.5:
            img_array = np.array(image)
            noise = np.random.normal(0, 5, img_array.shape)
            noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(noisy_img)
        
        # Randomly reduce image quality (increased from 75-95 to 85-98)
        if random.random() > 0.5:
            quality = random.randint(85, 98)
            temp_path = f"temp_{random.randint(0, 999999)}.jpg"
            try:
                image.save(temp_path, "JPEG", quality=quality)
                image = Image.open(temp_path)
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
        
        return image

    def generate_dataset(self, font_dir: str, samples_per_letter: int = 1000):
        """Generate the dataset using multiple fonts
        
        Args:
            font_dir (str): Directory containing STAM fonts
            samples_per_letter (int): Number of samples to generate per letter per font
        """
        # Clear and create output directory
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        
        # Load all fonts
        fonts = self.load_fonts(font_dir)
        print(f"Loaded {len(fonts)} font variations")
        
        # Generate samples for each letter with each font
        total_samples = len(self.hebrew_letters) * len(fonts) * samples_per_letter
        current_sample = 0
        
        for letter in self.hebrew_letters:
            # Create directory for this letter
            letter_dir = os.path.join(self.output_dir, letter)
            os.makedirs(letter_dir, exist_ok=True)
            
            for font, font_name in fonts:
                for i in range(samples_per_letter):
                    current_sample += 1
                    if current_sample % 100 == 0:
                        print(f"Generating sample {current_sample}/{total_samples}")
                    
                    # Generate base image
                    image = self.generate_letter_image(letter, font)
                    
                    # Apply random distortions
                    image = self.apply_random_distortions(image)
                    
                    # Save the image
                    filename = f"{letter}_{font_name}_{i:04d}.png"
                    image.save(os.path.join(letter_dir, filename))

if __name__ == "__main__":
    try:
        # Example usage
        generator = STAMDatasetGenerator("stam_dataset")
        
        # Check if font directory exists
        font_dir = "C:/Users/Owner/CascadeProjects/windsurf-project/mezuzah_ocr/fonts"
        if not os.path.exists(font_dir):
            print(f"Error: Font directory not found at {font_dir}")
            print("Please ensure you have installed STAM fonts and provided the correct path.")
            exit(1)
            
        # Generate dataset
        print(f"\nStarting dataset generation:")
        print(f"1. Looking for STAM fonts in: {font_dir}")
        print(f"2. Will generate variations in sizes: {', '.join(f'{k}={v}px' for k,v in generator.font_sizes.items())}")
        print(f"3. Output directory: {generator.output_dir}\n")
        
        generator.generate_dataset(
            font_dir=font_dir,
            samples_per_letter=1000
        )
        
        print("\nDataset generation complete!")
        print(f"Output directory: {os.path.abspath(generator.output_dir)}")
        
    except Exception as e:
        print(f"\nError during dataset generation: {str(e)}")
        exit(1)
