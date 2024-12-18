# -*- coding: utf-8 -*-
"""OCR.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14YOgSRx2Qj-LBLbFDPViMHLlOFlGeJcG
"""

# Installer
'''!sudo apt-get install tesseract-ocr
!pip install pytesseract
!pip install python-dateutil
from google.colab import drive
drive.mount('/content/drive')'''

import os
import json
import cv2
import numpy as np
from PIL import Image as Img
import pytesseract as pyt
import re
from dateutil import parser

pyt.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

desired_format = "%Y/%m/%d"
image_dir = "/content/drive/MyDrive/PaymentReceipt/dataset/val/KBZ"

# Regular expression patterns for extracting fields
transtype_pattern = re.compile(r"^(Transaction Type|Type)\s?:?\s?(.+)")
notes_pattern = re.compile(r"^(Notes|Note|Purpose|Reason)\s?:?\s?(.+)")
transtime_pattern = re.compile(r"^(Transaction Time|Date and Time|Date & Time|Transaction Date)\s?:?\s?(.+)")
transno_pattern = re.compile(r"^(Transaction No|Transaction ID)\s?:?\s?(.+)")
receiver_pattern = re.compile(r"^(To|Receiver Name|Send To)\s?:?\s?(.+)")
sender_pattern = re.compile(r"^(From|Sender Name|Send From)\s?:?\s?(.+)")
amount_data_pattern = re.compile(r"^(Amount|Total Amount)\s?:?\s?(.+)")

def extract_text_from_image(image_path):
    """
    Extracts text from an image using Tesseract OCR.

    :param image_path: Path to the image file
    :return: Extracted text as a string, or None if extraction fails
    """
    try:
        # Read the image using OpenCV
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian blur to reduce noise and smoothen the image
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Increase contrast using adaptive histogram equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(blurred)

        # Sharpen the image to make text more readable
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced_img, -1, kernel)

        # Apply a threshold to convert the image to binary (black and white)
        # Adjust the threshold value to ensure better extraction of gray text
        _, thresh = cv2.threshold(sharpened, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Inpainting to remove the watermark
        result = cv2.inpaint(img, thresh, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # Convert back to a PIL image
        pil_image = Img.fromarray(thresh)

        # Use Tesseract to do OCR on the image
        config = "--psm 6"
        text = pyt.image_to_string(pil_image, config=config, lang='eng')
        return text

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Split text into lines
def split_text_into_lines(text):
    lines = text.split('\n')
    return [line.strip() for line in lines if line.strip()]
        
def extract_date_time(date_time_str):
    """
    Extracts date and time from the input string using regex and dateutil parser.

    :param date_time_str: String containing date and time
    :return: Tdate, time
    """

    # Define regular expressions to match different date and time formats
    date_pattern = re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2} \w+ \d{4}|\w+ \d{1,2}, \d{4})")
    time_pattern = re.compile(r"\b((1[0-2]|0?[1-9]):[0-5][0-9](?::[0-5][0-9])?\s?[APap][Mm]|(2[0-3]|[01]?[0-9]):[0-5][0-9](?::[0-5][0-9])?)\b")  

    try:
          # Search date and time matches in the input string
          date_match = date_pattern.search(date_time_str)
          times_match = time_pattern.search(date_time_str) 
    
          # Parse the date part
          try:
              if date_match:
                date_obj = parser.parse(date_match.group())
                formatted_date = date_obj.strftime("%Y/%m/%d")
              else:
                formatted_date = ""
          except:
                formatted_date = ""
          
          # Parse the time part
          try:
              if times_match:
                time_obj = parser.parse(times_match.group())
                formatted_time = time_obj.strftime("%H:%M:%S")
              else:
                formatted_time = ""
          except:
                formatted_time = ""
        
    except Exception as e:
           print(f"Error parsing date or time: {e}")

    return formatted_date, formatted_time

def extract_amount_only(amount_str):    
    """
    Extracts numeric amount from the amount string using regex.

    :param amount_str: amount with negative sign, MMK, Ks
    :return: numeric amount as a string
    """

    formatted_amount = amount_str
    amount_only_pattern = re.compile(r"-?\d*(?:,\d*)*(?:\.\d{2})?")
    amount_pattern_match = amount_only_pattern.search(amount_str)
    
    if amount_pattern_match:
        return amount_pattern_match.group().replace("-","").strip()
        
    return amount_str

def extract_transaction_data(text):

    transaction_data = {
        "Transaction No" : None,
        "Date": None,
        "Time": None,
        "Transaction Time" : None,
        "Transaction Type": None,
        "Sender Name": None,
        "Amount": None,
        "Receiver Name": None,
        "Notes": None
    }
    lines = split_text_into_lines(text)
    for line in lines:
        # Transaction Time
        if re.search(transtime_pattern, line):
            transtime_pattern_match = transtime_pattern.search(line)
            date_time_str  = transtime_pattern_match.group(2).strip().strip('@').strip()
            transaction_data["Transaction Time"] = date_time_str            
            transaction_data["Date"], transaction_data["Time"] = extract_date_time(date_time_str)     

        # Transaction No
        elif re.search(transno_pattern, line):
             transno_pattern_match = transno_pattern.search(line)
             transaction_data["Transaction No"] = transno_pattern_match.group(2).strip().strip('@').strip()

        # Transaction Type
        elif re.search(transtype_pattern, line):
             transtype_pattern_match = transtype_pattern.search(line)
             transaction_data["Transaction Type"] = transtype_pattern_match.group(2).strip().strip('@').strip()

        # Amounts
        elif re.search(amount_data_pattern, line):
             amount_data_pattern_match = amount_data_pattern.search(line)
             amount_string = amount_data_pattern_match.group(2).strip().strip('@').strip()
             transaction_data["Amount"] = extract_amount_only(amount_string)

        # Sender Name
        elif re.search(sender_pattern, line):
             sender_pattern_match = sender_pattern.search(line)
             transaction_data["Sender Name"] = sender_pattern_match.group(2).strip().strip('@').strip()

        # Receiver Name
        elif re.search(receiver_pattern, line):
             receiver_pattern_match = receiver_pattern.search(line)
             transaction_data["Receiver Name"] = receiver_pattern_match.group(2).strip().strip('@').strip()

        # Notes
        elif re.search(notes_pattern, line):
            notes_match = notes_pattern.search(line)
            transaction_data["Notes"] = notes_match.group(2).strip()

    return transaction_data


# Process images and save extracted data to JSON
all_transactions = []

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, filename)

        try:
            # Extract text using Tesseract
            extracted_text = extract_text_from_image(image_path)
            print(f"Extracted data from {filename}")
            #print(f"Extracted data from {filename}: \n{extracted_text}\n")

            # Extract transaction information using regex
            transaction_info = extract_transaction_data(extracted_text)
            print(transaction_info)
            #transaction_info["File"] = filename  # Optional: Add filename for reference

            # Add to list of all transactions
            all_transactions.append(transaction_info)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

# Save the extracted transaction data to a JSON file
output_json_path = "transactions_data.json"
with open(output_json_path, 'w') as json_file:
    json.dump(all_transactions, json_file, indent=4)

print(f"All data saved to {output_json_path}")
