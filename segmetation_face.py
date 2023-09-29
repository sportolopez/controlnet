import cv2
import numpy as np
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn


url = "https://images.pexels.com/photos/733872/pexels-photo-733872.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"

def get_hair_segmentation(image):
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    pred_seg[pred_seg != 2] = 0
    arr_seg = pred_seg.cpu().numpy().astype("uint8")
    arr_seg *= 255

    pil_seg = Image.fromarray(arr_seg)
    pil_seg.save("segmentation.png")
    pil_seg.close()


image = Image.open("mujer1.PNG").convert("RGB")

get_hair_segmentation(image)




