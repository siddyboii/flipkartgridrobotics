import os
import cv2
import time
import csv
import re
import gradio as gr
from datetime import datetime
from PIL import Image
import torch
from ultralytics import YOLO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Model and directories
output_cropped_dir = 'saved_frames'  # Output directory for cropped frames
output_csv_path = 'inference_results.csv'  # CSV file for saving results
adapter_path = "newdescripterckp/checkpoint-241"
model_path = "best.pt"
threshold = 0.4

# Set up YOLO model
yolo_model = YOLO(model_path)

# Set up Qwen model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
#processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", max_pixels=1080*28*28)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", cache_dir="newdescripterckp", max_pixels=720*28*28)
model.load_adapter(adapter_path)

# Ensure output directories exist
if not os.path.exists(output_cropped_dir):
    os.makedirs(output_cropped_dir)

# Create or initialize CSV
if not os.path.exists(output_csv_path):
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sl no', 'Timestamp', 'Brand', 'Expiry date', 'Count', 'Expired', 'Expected life span (Days)'])

# Function to reduce bounding box
def reduce_bounding_box(x1, y1, x2, y2, reduction_factor=0.05):
    width = x2 - x1
    height = y2 - y1
    x1_new = x1 + int(reduction_factor * width)
    y1_new = y1 + int(reduction_factor * height)
    x2_new = x2 - int(reduction_factor * width)
    y2_new = y2 - int(reduction_factor * height)
    return x1_new, y1_new, x2_new, y2_new

# Function to calculate lifespan
def calculate_lifespan(expiry_date):
    try:
        expiry_datetime = datetime.strptime(expiry_date, "%m/%y")  # Parse expiry date
        current_datetime = datetime.now()
        lifespan = (expiry_datetime - current_datetime).days
        return lifespan if lifespan > 0 else "Expired"
    except ValueError:
        return "Invalid date"

# Enhanced Regex Patterns for Parsing
def parse_qwen_output(output_text):
    # Updated patterns
    brand_pattern = r"Brand:\s*([\w\s'\-]+)"  # Matches "Brand: Nestle" or "Brand: Nestle's"
    expiry_pattern = r"Expiry\s*Date:\s*(\d{1,2}/\d{2}(?:\d{2})?|(?:\d{1,2}\s)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4})"

    # Match brand
    brand_match = re.search(brand_pattern, output_text)
    brand = brand_match.group(1).strip() if brand_match else "Unknown"

    # Match expiry date
    expiry_match = re.search(expiry_pattern, output_text)
    expiry_date = expiry_match.group(1).strip() if expiry_match else "Unknown"

    # Normalize expiry date for consistent processing
    if expiry_date != "Unknown":
        expiry_date = normalize_expiry_date(expiry_date)

    return brand, expiry_date


# Normalize Expiry Date for Consistent Lifespan Calculation
def normalize_expiry_date(expiry_date):
    try:
        # Handle MM/YY and MM/YYYY
        if re.match(r"^\d{1,2}/\d{2,4}$", expiry_date):
            if len(expiry_date.split("/")[-1]) == 2:  # MM/YY
                return datetime.strptime(expiry_date, "%m/%y").strftime("%m/%y")
            else:  # MM/YYYY
                return datetime.strptime(expiry_date, "%m/%Y").strftime("%m/%Y")

        # Handle DD/MM/YY or DD/MM/YYYY
        elif re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", expiry_date):
            if len(expiry_date.split("/")[-1]) == 2:  # DD/MM/YY
                return datetime.strptime(expiry_date, "%d/%m/%y").strftime("%m/%y")
            else:  # DD/MM/YYYY
                return datetime.strptime(expiry_date, "%d/%m/%Y").strftime("%m/%Y")

        # Handle written formats (e.g., "December 2023", "12 Dec 2023")
        elif re.match(r"^(?:\d{1,2}\s)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4}$", expiry_date, re.IGNORECASE):
            return datetime.strptime(expiry_date, "%d %b %Y").strftime("%m/%Y") if expiry_date[0].isdigit() else datetime.strptime(expiry_date, "%b %Y").strftime("%m/%Y")

        return "Invalid Format"  # If no format matches
    except ValueError:
        return "Invalid Format"


# Function to append to CSV
def append_to_csv(timestamp, brand, expiry_date, object_count):
    lifespan = calculate_lifespan(expiry_date)
    expired = "Yes" if lifespan == "Expired" else "No"
    with open(output_csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([object_count, timestamp, brand, expiry_date, 1, expired, lifespan])

# Function to process each object and run Qwen inference
def process_cropped_object(cropped_pil_image, frame_count, object_count):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": cropped_pil_image},
                {"type": "text", "text": "Identify the brand name, product type, expiry date, manufacturing date, quantity only."}
            ]
        }
    ]

    # Prepare input for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate output from the Qwen model
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Parse the output and append to CSV
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    brand, expiry_date = parse_qwen_output(output_text)
    append_to_csv(timestamp, brand, expiry_date, object_count)

    return output_text

# Gradio function to process live feed
def process_live_feed_with_csv():
    cap = cv2.VideoCapture(1)  # Open webcam feed (0 for default camera)
    if not cap.isOpened():
        return "Error: Could not access webcam."

    inference_results = ""
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        object_count = 0
        current_inference = ""

        # Run YOLO on the frame and process detected objects
        results = yolo_model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                object_count += 1

                # Reduce bounding box
                x1_new, y1_new, x2_new, y2_new = reduce_bounding_box(x1, y1, x2, y2)
                cropped_image = frame[int(y1_new):int(y2_new), int(x1_new):int(x2_new)]
                cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

                # Run Qwen inference and collect results
                current_inference += process_cropped_object(cropped_pil_image, frame_count, object_count) + "\n"

        inference_results += current_inference

        # Update the Gradio UI
        yield gr.update(value=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), gr.update(value=inference_results)

    cap.release()
    cv2.destroyAllWindows()

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Live Video Inference with YOLO and Qwen")

    video_output = gr.Video(label="Live Video Feed")
    output_textbox = gr.Textbox(label="Inference Output", interactive=False)

    submit_button = gr.Button("Start Inference")
    submit_button.click(process_live_feed_with_csv, inputs=None, outputs=[video_output, output_textbox])

# Launch Gradio UI
demo.launch(share=True)

