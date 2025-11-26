import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.device_utils import DEVICE

# Load the model
model = build_sam3_image_model()
model.to(DEVICE)
processor = Sam3Processor(model)

# Load an image
img_path = "/Users/dkneubuhler/Downloads/1597009.png"
prompt = "eyes and mouth"

image = Image.open(img_path)
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt=prompt)

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

print(f"{masks=}")