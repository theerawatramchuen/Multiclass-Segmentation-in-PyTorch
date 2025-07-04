import os
import cv2
import numpy as np
import torch
from model import build_unet
from utils import create_dir
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*weights_only.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")

# Configuration
IMAGE_SIZE = (256, 256)
COLORMAP = [
    [0, 0, 0],      # Background (black)
    [0, 0, 128],    # Weed-1 (dark blue)
    [0, 128, 0]     # Weed-2 (green)
]
CHECKPOINT_PATH = "files/checkpoint.pth"
INPUT_DIR = "your_images/"
OUTPUT_DIR = "inference_results/"

def index_to_rgb_mask(mask, colormap):
    """Convert class index mask to RGB mask"""
    height, width = mask.shape
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, rgb in enumerate(colormap):
        rgb_mask[mask == class_id] = rgb
    return rgb_mask

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_unet(num_classes=len(COLORMAP)).to(device)

# Safe model loading with weights_only parameter
try:
    # Try with weights_only=True for newer PyTorch versions
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
except TypeError:
    # Fallback for older PyTorch versions
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Create output directories
create_dir(OUTPUT_DIR)
create_dir(f"{OUTPUT_DIR}/overlays")
create_dir(f"{OUTPUT_DIR}/masks")

print(f"Starting inference on images in: {INPUT_DIR}")
for image_name in os.listdir(INPUT_DIR):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    image_path = os.path.join(INPUT_DIR, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        continue

    original_h, original_w = image.shape[:2]
    print(f"Processing: {image_name} ({original_w}x{original_h})")
    
    # Preprocess image
    resized_image = cv2.resize(image, IMAGE_SIZE)
    x = resized_image / 255.0
    x = np.transpose(x, (2, 0, 1))  # HWC to CHW
    x = torch.from_numpy(x).unsqueeze(0).float().to(device)

    # Run inference
    with torch.no_grad():
        pred = model(x)
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = pred.squeeze().cpu().numpy().astype(np.uint8)

    # Convert prediction to RGB mask
    pred_mask = index_to_rgb_mask(pred, COLORMAP)
    
    # Resize mask to original dimensions
    pred_mask_orig = cv2.resize(
        pred_mask, 
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )

    # Create overlay
    overlay = image.copy()
    alpha = 0.5
    
    # Process each class (skip background)
    for class_idx in range(1, len(COLORMAP)):
        class_color = COLORMAP[class_idx]
        # Create mask for current class
        class_mask = np.all(pred_mask_orig == class_color, axis=-1)
        
        # Create colored image for this class
        color_overlay = np.zeros_like(image)
        color_overlay[class_mask] = class_color
        
        # Blend with original image
        overlay = cv2.addWeighted(overlay, 1, color_overlay, alpha, 0)

    # Save results
    base_name = os.path.splitext(image_name)[0]
    
    # Save mask
    mask_path = f"{OUTPUT_DIR}/masks/{base_name}_mask.png"
    cv2.imwrite(mask_path, pred_mask_orig)
    
    # Save overlay
    overlay_path = f"{OUTPUT_DIR}/overlays/{base_name}_overlay.png"
    cv2.imwrite(overlay_path, overlay)
    
    print(f"Saved results for {image_name}")

print("\nInference complete! All results saved to:", OUTPUT_DIR)