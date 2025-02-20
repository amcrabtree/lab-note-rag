import openai
import os
import base64
from pdf2image import convert_from_path
from PIL import Image
import io
import argparse


default_prompt = """
    You are a microbiology researcher who is tasked with transcribing all the handwritten notes 
    from images of lab notebook pages. Extract all handwritten text from this lab notebook page 
    and return it as markdown text. 
    Bold any text that looks like a date. 
    Italicise any text in Latin. 
    If a date looks like the start of a new notebook entry, then make it a header. 
    if you detect a hand-drawn diagram, write a hash tag "#diagram" for each diagram detected. 
    If you see a page number at the very bottom of the page, make it the main header for this page.
    """


def pdf_to_image_list(pdf_path, dpi=150):
    pil_img_list = convert_from_path(pdf_path, dpi)
    return pil_img_list


def encode_image_from_file(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_pil_image(pil_image: Image):
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode("utf-8")


def extract_text_from_image_openai(pil_image: Image, client: openai.OpenAI, prompt: str):
    """Extract handwritten text from an image using OpenAI GPT API"""
    image_data = encode_pil_image(pil_image) 

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  
            messages=[
                {"role": "user", 
                 "content": [
                     {"type": "text", 
                      "text": prompt},
                     {"type": "image_url", 
                      "image_url": {
                          "url": f"data:image/png;base64,{image_data}"
                      }
                     }]
                }
            ],
            max_tokens=2000
        )
    
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def convert_notebook_to_text(pdf_path: str, api_key: str, prompt: str):
    # Set up OpenAI client
    client = openai.OpenAI(api_key=api_key)

    # Extract PIL images from PDF
    pil_img_list = pdf_to_image_list(pdf_path)

    # Extract text from each image
    final_md_text = ""
    for pil_img in pil_img_list:
        text_output = extract_text_from_image_openai(pil_img, client, prompt)
        final_md_text += (text_output + "\n\n")
    return final_md_text


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Turn PDF lab notebooks into Markdown.")
parser.add_argument("--notebook", "-n",
                    help="Path to notebook PDF file.")
parser.add_argument("--output_file", "-o", 
                    help="Path to destination output text file.")
parser.add_argument("--api_key", "-a", 
                    help="Your OpenAI API key used for billing.")
parser.add_argument("--prompt", "-p", default="",
                    help="Custom prompt used to extract notebook. Optional.")

if __name__ == "__main__":
    args = parser.parse_args()
    
    prompt = args.prompt if args.prompt != "" else default_prompt
    notebook_text = convert_notebook_to_text(args.notebook, args.api_key, args.prompt)

    with open(args.output_file, "w") as f:
        f.write(notebook_text)

    print(f"Text extraction complete. Output saved to {args.output_file}")
