import os
import cv2
import time
import gradio as gr
from PIL import Image
import torch
from ultralytics import YOLO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Model and directories
output_cropped_dir = 'saved_frames'  # Output directory for cropped frames
#adapter_path = "/teamspace/studios/this_studio/newdescripterckp/checkpoint-241"
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
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", max_pixels=1080*28*28)

# Ensure output directories exist
if not os.path.exists(output_cropped_dir):
    os.makedirs(output_cropped_dir)

# Function to reduce bounding box and mask
def reduce_bounding_box(x1, y1, x2, y2, reduction_factor=0.05):
    width = x2 - x1
    height = y2 - y1
    x1_new = x1 + int(reduction_factor * width)
    y1_new = y1 + int(reduction_factor * height)
    x2_new = x2 - int(reduction_factor * width)
    y2_new = y2 - int(reduction_factor * height)
    return x1_new, y1_new, x2_new, y2_new

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
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # Format the output nicely for each object
    formatted_output = (
        f"Inference Output for Frame {frame_count}, Object {object_count}:\n"
        f"----------------------------------------------------\n"
        f"{output_text[0]}\n"
        f"----------------------------------------------------\n"
    )

    # Print the output after every inference
    print(formatted_output)

    return formatted_output

# Gradio function to process live feed and display results dynamically
def process_live_feed_with_threshold():
    cap = cv2.VideoCapture(1)  # Open webcam feed (0 for default camera)
    if not cap.isOpened():
        return "Error: Could not access webcam."

    inference_results = ""
    frame_count = 0
    last_detection_time = time.time()
    detection_occurred = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        object_count = 0  # Reset object count for the current frame
        current_inference = ""

        # Run YOLO on the frame and process detected objects
        results = yolo_model(frame)[0]
        current_frame_detections = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                object_count += 1
                current_frame_detections.append((x1, y1, x2, y2))

                # Reduce bounding box
                x1_new, y1_new, x2_new, y2_new = reduce_bounding_box(x1, y1, x2, y2)
                cropped_image = frame[int(y1_new):int(y2_new), int(x1_new):int(x2_new)]
                cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

                # Run Qwen inference and collect results
                current_inference += process_cropped_object(cropped_pil_image, frame_count, object_count)

        # Check if detections occurred
        if current_frame_detections:
            detection_occurred = True
            last_detection_time = time.time()  # Update last detection time
        else:
            # If no detection for the threshold time, process last frame
            if detection_occurred and time.time() - last_detection_time > detection_threshold_time:
                detection_occurred = False  # Reset flag
                for x1, y1, x2, y2 in current_frame_detections:
                    cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
                    cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

                    # Run Qwen inference for the saved detections
                    current_inference += process_cropped_object(cropped_pil_image, frame_count, object_count)

        # Append current inference results to the overall results
        inference_results += current_inference

        # Update the Gradio UI
        yield gr.update(value=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), gr.update(value=inference_results)

    cap.release()
    cv2.destroyAllWindows()

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Live Video Inference with YOLO and Qwen (Time-Threshold-Based Processing)")

    video_output = gr.Video(label="Live Video Feed")
    output_textbox = gr.Textbox(label="Inference Output", interactive=False)

    submit_button = gr.Button("Start Inference")
    submit_button.click(process_live_feed_with_threshold, inputs=None, outputs=[video_output, output_textbox])

# Launch Gradio UI
demo.launch(share=True)
