import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import os
from models.resnet_unet import ResNetUNet 

def predict(image_path, model_path, output_mask_path, output_overlay_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ResNetUNet(n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load original image for size reference
    image_pil = Image.open(image_path).convert("RGB")
    original_size = image_pil.size  # (width, height)

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output)[0, 0].cpu().numpy()

    # Resize prediction mask back to original size
    pred_resized = cv2.resize(pred, original_size, interpolation=cv2.INTER_LINEAR)
    binary_mask = (pred_resized > 0.2).astype(np.uint8) * 255

    # Overlay on original image
    orig_np = np.array(image_pil)
    color_mask = np.zeros_like(orig_np)
    color_mask[:, :, 2] = binary_mask  # Red channel
    overlay = cv2.addWeighted(orig_np, 1.0, color_mask, 0.5, 0)

    # Save results
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    cv2.imwrite(output_mask_path, binary_mask)
    cv2.imwrite(output_overlay_path, overlay)

    print(f"Mask saved to     : {output_mask_path}")
    print(f"Overlay saved to  : {output_overlay_path}")

if __name__ == "__main__":
    predict(
        image_path="data/sample.jpg",
        model_path="weights/model.pth",
        output_mask_path="output/mask.png",
        output_overlay_path="output/overlay.png"
    )
