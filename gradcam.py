import os
import time
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.nn.functional as F

from model import build_unet
from utils import create_dir, seeding
from train import load_data


def get_conv_layer(model, conv_layer_name):
    for name, layer in model.named_modules():
        if name == conv_layer_name:
            return layer
    raise ValueError(f"Layer '{conv_layer_name}' not found in the model.")


def compute_segmentation_gradcam(model, image_tensor, target_class, conv_layer_name=None):
    ## Finds the layer
    conv_layer = get_conv_layer(model, conv_layer_name)
    activations = []
    gradients = []

    ## Define hooks
    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    ## Forward hook saves output feature maps of the chosen layer.
	## Backward hook captures the gradients of the layer’s output during backpropagation.

    forward_handle = conv_layer.register_forward_hook(forward_hook)
    backward_handle = conv_layer.register_backward_hook(backward_hook)

    ## Make prediction and get class-specific score
    output = model(image_tensor)
    probs = torch.softmax(output, dim=1)
    target = probs[0, target_class, :, :].mean()

    ## Backpropagate to compute gradients
    model.zero_grad()
    target.backward()

    grads_val = gradients[0].detach()[0]
    acts_val = activations[0].detach()[0]

    ## Compute channel-wise weights and generate Grad-CAM
    weights = grads_val.mean(dim=(1, 2))
    gradcam = torch.zeros(acts_val.shape[1:], dtype=torch.float32).to(image_tensor.device)
    for i, w in enumerate(weights):
        gradcam += w * acts_val[i]

    ## Generate Grad-CAM map
    gradcam = F.relu(gradcam)
    gradcam = gradcam - gradcam.min()
    gradcam = gradcam / (gradcam.max() + 1e-8)
    gradcam = gradcam.cpu().numpy()

    ## Clean up: remove hooks
    forward_handle.remove()
    backward_handle.remove()
    return gradcam

def overlay_heatmap_on_image(image_np, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image_np, alpha, heatmap, 1 - alpha, 0)
    return superimposed_img

def apply_gradcam(model, save_path, test_x, test_y, size, colormap, layer=None):
    for i, (x_path, y_path) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = os.path.basename(x_path).split(".")[0]

        # Input image
        image = cv2.imread(x_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image.copy()

        input_image = np.transpose(image, (2, 0, 1)) / 255.0
        input_image = np.expand_dims(input_image, axis=0).astype(np.float32)
        input_tensor = torch.from_numpy(input_image).to(device)

        # Read and resize ground-truth mask
        mask = cv2.imread(y_path, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, size)

        for class_idx in range(len(colormap)):
            # Compute Grad-CAM
            gradcam = compute_segmentation_gradcam(model, input_tensor, class_idx, conv_layer_name=layer)
            cam_img = overlay_heatmap_on_image(save_img.copy(), gradcam)

            # Create binary mask for that class
            class_rgb = np.array(colormap[class_idx], dtype=np.uint8)
            binary_mask = cv2.inRange(mask, class_rgb, class_rgb)
            binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

            # Concatenate: Original | Grad-CAM | Binary Mask
            line = np.ones((size[1], 10, 3), dtype=np.uint8) * 255
            combined_img = np.concatenate([save_img, line, cam_img, line, binary_mask], axis=1)

            # Save
            cam_dir = f"{save_path}/gradcam/{class_idx}"
            os.makedirs(cam_dir, exist_ok=True)
            cv2.imwrite(f"{cam_dir}/{name}.jpg", combined_img)

if __name__ == "__main__":
    seeding(42)

    image_w = 256
    image_h = 256
    size = (image_w, image_h)
    dataset_path = "./Weeds-Dataset/weed_augmented"

    colormap = [
        [0, 0, 0],      # Background
        [0, 0, 128],    # Weed-1
        [0, 128, 0]     # Weed-2
    ]
    num_classes = len(colormap)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unet(num_classes=num_classes)
    model = model.to(device)

    checkpoint_path = "files/checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    model.eval()
    
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)
    test_x, test_y = test_x[:10], test_y[:10]  # Limit to 10 samples for demonstration

    save_path = "results"
    create_dir(f"{save_path}/gradcam")

    apply_gradcam(model, save_path, test_x, test_y, size, colormap, layer="d4.conv.conv.5")

    ## Example of layer names that can be used
    """
    e1.conv.conv.5
    e2.conv.conv.5
    e3.conv.conv.5
    e4.conv.conv.5
    b.conv.5
    d1.conv.conv.5
    d2.conv.conv.5
    d3.conv.conv.5
    d4.conv.conv.5
    """