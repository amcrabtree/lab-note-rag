import os
from pdf2image import convert_from_path
import cv2
import pytesseract
from PIL import Image


def pdf_to_images(pdf_path, dpi=300):
    return convert_from_path(pdf_path, dpi)

def preprocess_image(image):
    # Convert PIL image to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)

    return denoised

def extract_text_from_image(image):
    custom_config = r'--oem 1 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text

def process_pdf(pdf_path, output_text_path):
    images = pdf_to_images(pdf_path)
    all_text = ""

    for i, image in enumerate(images):
        print(f"Processing page {i+1}...")
        preprocessed_img = preprocess_image(image)
        text = extract_text_from_image(preprocessed_img)
        all_text += f"\n--- Page {i+1} ---\n{text}"

    with open(output_text_path, 'w', encoding='utf-8') as f:
        f.write(all_text)

    print(f"Text extraction complete. Output saved to {output_text_path}")

if __name__ == "__main__":
    import numpy as np

    # Input and output paths
    pdf_path = '/Users/amc/Downloads/old_downloads/lab_notebooks/sample_notebook.pdf'
    output_text_path = '/Users/amc/Downloads/old_downloads/lab_notebooks/extracted_text.txt'

    process_pdf(pdf_path, output_text_path)
