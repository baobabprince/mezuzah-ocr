# Mezuzah Kosher OCR Verification Tool

## Project Overview
This application is designed to verify the kosher status of mezuzah scrolls using advanced OCR and machine learning techniques specifically trained on Hebrew STAM (Sifrei Torah, Tefillin, and Mezuzot) fonts.

## Features
- Specialized OCR for Hebrew STAM fonts
- Letter-by-letter kosher validation
- Machine learning-based letter recognition
- User-friendly interface for scroll inspection

## Setup Instructions
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download pre-trained model (if available)
3. Run the application:
   ```
   python app.py
   ```

## Halachic Considerations
This tool is intended to assist sofrim (scribes) but should NOT replace a qualified rabbinic inspection. Always consult a Torah expert for final kosher determination.

## Data Privacy
No scroll images are stored. All processing is done in-memory with immediate deletion.
