# flipkartgridrobotics

VIDEO_SUBMISSION_WITH_EXPLAINATION [here](https://drive.google.com/drive/folders/1TlvhGWoAjPB170CmmnP5PZn4FtJTxol0?usp=sharing).


This repository implements state-of-the-art AI techniques for produce freshness detection and real-time product descriptor identification. It utilizes models like Qwen-2B and a fine-tuned CLIP for efficient processing.

## Files Overview

### 1. **Freshness Prediction Classification** (`freshness_prediction_classificationv2.ipynb`)
- **Purpose**: Classifies produce as **fresh** or **rotten** using a fine-tuned CLIP model.
- **Use Case**: Bulk image datasets for inventory management.

### 2. **Live Produce Freshness Detection** (`live_produce_freshness_v2.ipynb`)
- **Purpose**: Detects freshness levels such as **extreme freshness**, **mild freshness**, or **worse** in real-time using Qwen-2B.
- **Use Case**: Live webcam feeds for detecting freshness of produce.
- **Outputs**: Produce, Freshness index, shelf life estimation.

### 3. **Live Product Descriptor** (`live_product_descriptor_v2.py`)
- **Purpose**: Extracts product details like **brand name** and **expiry date** in real-time from conveyor belt setups using Qwen-2B.
- **Use Case**: Real time Product Information retrieval.
- **Outputs**: Brand Name, Expiry Date, Item Count, Expiry status and remaining lifespan in days.

## Technologies Used
- **YOLO**: Object detection.
- **Qwen-2B**: Image and text understanding.
- **Fine-tuned CLIP**: Static image classification.
- **Python Libraries**: `torch`, `transformers`, `cv2`, `csv`, `Pillow`.
