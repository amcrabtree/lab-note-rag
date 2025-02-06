import os
from pdf2image import convert_from_path
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# Load TrOCR model and processor
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

def pdf_to_images(pdf_path, dpi=300):
    return convert_from_path(pdf_path, dpi)

def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)

    return denoised

def detect_and_mask_diagrams(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # Assume diagrams are large and irregular; adjust area/ratio as needed
        if area > 5000 and (w/h > 1.5 or h/w > 1.5):
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)

    return image

def extract_text_from_image(image):
    pil_image = Image.fromarray(image).convert('RGB') 
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

def process_pdf(pdf_path, output_text_path):
    images = pdf_to_images(pdf_path)
    all_text = ""

    for i, image in enumerate(images):
        print(f"Processing page {i+1}...")
        preprocessed_img = preprocess_image(image)
        masked_img = detect_and_mask_diagrams(preprocessed_img)
        text = extract_text_from_image(masked_img)
        all_text += f"\n--- Page {i+1} ---\n{text}"

    with open(output_text_path, 'w', encoding='utf-8') as f:
        f.write(all_text)

    print(f"Text extraction complete. Output saved to {output_text_path}")

if __name__ == "__main__":
    # Input and output paths
    pdf_path = '/Users/amc/Downloads/old_downloads/lab_notebooks/sample_notebook.pdf'
    output_text_path = '/Users/amc/Downloads/old_downloads/lab_notebooks/extracted_text.txt'

    process_pdf(pdf_path, output_text_path)
