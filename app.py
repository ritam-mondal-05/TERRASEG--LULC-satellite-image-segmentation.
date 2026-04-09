import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import time
import base64
import io
import re

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="TerraSeg — LULC Framework",
    page_icon="🛰️",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    padding-left: 0rem !important;
    padding-right: 0rem !important;
    max-width: 100% !important;
}

/* Custom Sci-Fi Scrollbar for entire Streamlit window */
::-webkit-scrollbar {
    width: 12px;
}
::-webkit-scrollbar-track {
    background: #020305;
    border-left: 1px solid rgba(255,255,255,0.05);
}
::-webkit-scrollbar-thumb {
    background: rgba(110,193,228,0.3);
    border-radius: 6px;
    border: 3px solid #020305;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(132,41,246,0.8);
}
</style>
""", unsafe_allow_html=True)

# 2. COLOR TAXONOMY
# Order: Water, Land, Road, Building, Vegetation, Unlabeled
HEX_COLORS = ['#E2A929', '#8429F6', '#6EC1E4', '#3C1098', '#FEDD3A', '#9B9B9B']
class_names = ["Water", "Land", "Road", "Building", "Vegetation", "Unlabeled"]

# Helper to convert Hex to RGB for mask generation
def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return [int(hex_str[i:i+2], 16) for i in (0, 2, 4)]

# 3. LOAD THE MODEL
@st.cache_resource
def load_segmentation_model():
    model_path = "satellite_segmentation_full.keras"
    if os.path.exists(model_path):
        # compile=False avoids errors with custom loss/metrics like Jaccard
        return tf.keras.models.load_model(model_path, compile=False)
    return None

model = load_segmentation_model()

# 4. LOAD FRONTEND
def get_html_content():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Error: index.html not found!</h1>"

# 5. SIDEBAR & STATUS
st.sidebar.title("🛰️ Control Panel")

if model:
    st.sidebar.success("● Model Loaded: .keras (Active)")
else:
    st.sidebar.error("○ Model Status: Offline (Check file path)")

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Satellite Image", type=["jpg", "jpeg", "png", "tif"])

# 6. MAIN LOGIC
html_raw = get_html_content()

if uploaded_file is None:
    # --- IDLE STATE ---
    # Show the default HTML dashboard (with demo data)
    components.html(html_raw, height=1000, scrolling=True)
    st.info("Please upload a satellite image from the sidebar to begin analysis.")
else:
    # --- ACTIVE STATE ---
    st.sidebar.warning("⚡ Model Engine: Processing...")
    
    # 1. Preprocessing
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    # Resize to match model input (256x256)
    img_resized = cv2.resize(img_array, (256, 256))
    img_input = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # 2. Inference
    start_time = time.time()
    prediction = model.predict(img_input)
    # Get the class with highest probability for each pixel
    mask = np.argmax(prediction, axis=3)[0] 
    inference_time = time.time() - start_time

    # 3. Inject Actual Data into HTML Dashboard
    # Resize mask to 16x16 for the mini-grid visualization in your HTML (fallback)
    small_mask = cv2.resize(mask.astype(np.uint8), (16, 16), interpolation=cv2.INTER_NEAREST)
    grid_data = small_mask.tolist()

    # Pre-calculate stats for HTML injection
    h, w = mask.shape
    u, counts = np.unique(mask, return_counts=True)
    total_pixels = h * w

    # Prepare high-res base64 image for scanner injection
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i, hex_color in enumerate(HEX_COLORS):
        rgb = hex_to_rgb(hex_color)
        color_mask[mask == i] = rgb
            
    img_pil = Image.fromarray(color_mask)
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    scan_html = f'''
    <div class="scan-container" style="position:relative;width:100%;height:100%;aspect-ratio:1;overflow:hidden;border-radius:12px;">
        <img src="data:image/png;base64,{img_str}" style="width:100%;height:100%;object-fit:cover;">
    </div>
    '''

    injected_html = html_raw.replace(
        "fillCanvas('demoSeg', 16);", 
        f"updateRealGrid('demoSeg', {grid_data});"
    )
    
    # Replace the HTML demo box with the HD base64 image
    injected_html = re.sub(r'<div id="demoSeg"[^>]*>.*?</div>', scan_html, injected_html, flags=re.DOTALL)

    # Inject real metric percentages
    for i in range(6):
        px_count = dict(zip(u, counts)).get(i, 0)
        percentage = (px_count / total_pixels) * 100
        injected_html = injected_html.replace(f"[PCT_{i}]", f"{percentage:.1f}")

    # Convert the Input Image to Base64
    buffered_in = io.BytesIO()
    image.save(buffered_in, format="JPEG")
    input_b64 = base64.b64encode(buffered_in.getvalue()).decode()

    # Generate the Class Distribution HTML Block
    stats_html = ""
    for i in range(6):
        px_count = dict(zip(u, counts)).get(i, 0)
        percentage = (px_count / total_pixels) * 100
        stats_html += f'''
        <div style="background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:16px;text-align:center;" class="tilt-card">
            <div style="font-size:0.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">{class_names[i]}</div>
            <div style="font-size:1.5rem;font-weight:700;color:white;margin-bottom:12px;">{percentage:.1f}%</div>
            <div style="background:{HEX_COLORS[i]};height:6px;border-radius:3px;box-shadow:0 0 10px {HEX_COLORS[i]}"></div>
        </div>
        '''

    # Build the complete Deep Learning Output HTML
    dl_output_html = f'''
    <div class="section" id="dl-output" style="padding: 40px !important;">
      <div class="section-label">Results</div>
      <h2 class="section-title">Deep Learning Analysis Output</h2>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:40px;">
         <div class="result-panel tilt-card">
           <div class="result-panel-header"><span>Input Satellite Imagery</span></div>
           <div class="scan-container" style="position:relative;width:100%;aspect-ratio:1;overflow:hidden;border-radius:0 0 8px 8px;">
               <img src="data:image/jpeg;base64,{input_b64}" style="width:100%;height:100%;object-fit:cover;">
           </div>
         </div>
         <div class="result-panel tilt-card">
           <div class="result-panel-header"><span>Predicted Segmentation Mask</span></div>
           <div style="position:relative;width:100%;aspect-ratio:1;overflow:hidden;border-radius:0 0 8px 8px;">
             <img src="data:image/png;base64,{img_str}" style="width:100%;height:100%;object-fit:cover;">
           </div>
         </div>
      </div>
      <h3 style="color:white;font-family:var(--font-main);font-size:1.2rem;margin-bottom:20px;">Class Distribution</h3>
      <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:16px;">
         {stats_html}
      </div>
      <hr style="border:0;border-top:1px solid rgba(255,255,255,0.05);margin-top:40px;">
    </div>
    '''

    # Inject the constructed DL UI into the iframe before the architecture section
    injected_html = injected_html.replace("<!-- DL_OUTPUT_MARKER -->", dl_output_html)

    # Replace the placeholder inference time in the HTML
    injected_html = injected_html.replace("0.38s", f"{inference_time:.2f}s")
    
    # Render the updated Dashboard (Increased height slightly to accommodate the new native outputs)
    components.html(injected_html, height=2200, scrolling=True)

# Reset Button
if st.sidebar.button("Clear Cache & Reset"):
    st.cache_resource.clear()
    st.rerun()