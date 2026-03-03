import gradio as gr
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import cv2
import os

# Load your trained model (adjust path)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.load_state_dict(torch.load("siam_unet_levir_cdplus.pt", map_location=device))
model.to(device)
model.eval()

def predict_full(a_pil, b_pil, thr=0.5):
    """Core prediction function (your existing logic)"""
    a = TF.to_tensor(a_pil).unsqueeze(0).to(device)
    b = TF.to_tensor(b_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(a,b))[0,0].cpu().numpy()
    mask = (prob > thr).astype(np.uint8)
    return prob, mask

def contour_overlay(b_pil, mask_u8):
    """Day30 with red contour marking"""
    b = np.array(b_pil)
    contours, _ = cv2.findContours(mask_u8 * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = b.copy()
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness=4)
    return Image.fromarray(overlay)

def change_stats(mask):
    changed_px = mask.sum()
    total_px = mask.size
    pct = 100 * changed_px / total_px
    return f"{changed_px:,} px ({pct:.1f}%)"

def detect_changes(day1_img, day30_img, threshold=0.5):
    """Main UI function: Day1 + Day30 → change overlay"""
    if day1_img is None or day30_img is None:
        return None, None, "Upload both Day1 and Day30 images"
    
    # Predict
    _, mask = predict_full(day1_img, day30_img, thr=float(threshold))
    
    # Generate outputs
    overlay = contour_overlay(day30_img, mask)
    stats_text = change_stats(mask)
    
    return day1_img, overlay, stats_text

# Create Gradio interface
with gr.Blocks(title="Change Detection") as demo:
    gr.Markdown("# 🛰️ Satellite Change Detection")
    gr.Markdown("Upload **Day 1** and **Day 30** images to see changes highlighted in **red contours**.")
    
    with gr.Row():
        with gr.Column(scale=1):
            day1_input = gr.Image(type="pil", label="Day 1 Image")
            day30_input = gr.Image(type="pil", label="Day 30 Image")
            threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Change threshold")
            detect_btn = gr.Button("Detect Changes", variant="primary")
        
        with gr.Column(scale=1):
            day1_output = gr.Image(label="Day 1")
            overlay_output = gr.Image(label="Changes (red contours)")
            stats_output = gr.Textbox(label="Change Stats")
    
    # Connect inputs → outputs
    detect_btn.click(
        fn=detect_changes,
        inputs=[day1_input, day30_input, threshold],
        outputs=[day1_output, overlay_output, stats_output]
    )

if __name__ == "__main__":
    demo.launch(
        share=True,  # generates public URL
        server_name="0.0.0.0",  # accessible from network
        server_port=7860
    )
