import streamlit as st
import numpy as np
import cv2
from deepforest import main
from shapely.geometry import box

# === Streamlit App ===
st.title("Tree Canopy Detection & Clearing Cost Estimator 🌳")

# Sidebar for settings
st.sidebar.header("Settings")
altitude_ft = st.sidebar.number_input("Altitude (ft)", value=300.0)
sensor_width_mm = st.sidebar.number_input("Sensor width (mm)", value=13.2)
focal_length_mm = st.sidebar.number_input("Focal length (mm)", value=8.8)
cost_per_m2 = st.sidebar.number_input("Cost per m² to clear", value=5.0)

# Smoothing parameters — you can expose these in sidebar if desired
DILATE_SIZE = 21
CLOSE_SIZE = 21
GAUSS_KSIZE = 21
GAUSS_SIGMA = 7
THRESH = 127
ALPHA = 0.35

uploaded_image = st.file_uploader("Upload an aerial image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Read image
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    img_height, img_width = image.shape[:2]

    # === Load DeepForest Model ===
    model = main.deepforest()
    model.use_release()

    # Run prediction
    predictions = model.predict_image(image)

    # Filter predictions by confidence score
    confidence_threshold = 0.19  # try 0.3, 0.5, 0.7, etc.
    predictions = predictions[predictions["score"] >= confidence_threshold]

    total_tree_area_px = 0
    polygons = []

    for _, row in predictions.iterrows():
        xmin, ymin, xmax, ymax = row[["xmin", "ymin", "xmax", "ymax"]]
        total_tree_area_px += (xmax - xmin) * (ymax - ymin)
        polygons.append(box(xmin, ymin, xmax, ymax))
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)


    # === Pixel resolution (GSD) calculation ===
    altitude_m = altitude_ft * 0.3048
    pixel_resolution_m = (sensor_width_mm * altitude_m) / (focal_length_mm * img_width)

    # Create empty mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Fill mask with predicted bounding boxes
    for _, row in predictions.iterrows():
        xmin, ymin, xmax, ymax = map(int, row[["xmin", "ymin", "xmax", "ymax"]])
        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, thickness=-1)


    # === Smoothing using morphological operations ===
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_SIZE, DILATE_SIZE))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_SIZE, CLOSE_SIZE))

    mask_d = cv2.dilate(mask, dilate_kernel, iterations=1)
    mask_dc = cv2.morphologyEx(mask_d, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    if GAUSS_KSIZE > 0:
        mask_blur = cv2.GaussianBlur(mask_dc, (GAUSS_KSIZE, GAUSS_KSIZE), GAUSS_SIGMA)
        _, mask_final = cv2.threshold(mask_blur, THRESH, 255, cv2.THRESH_BINARY)
    else:
        mask_final = mask_dc.copy()

    merged_area_px = int(cv2.countNonZero(mask_final))
    merged_area_m2 = merged_area_px * (pixel_resolution_m ** 2)

    # === Cost estimation ===
    estimated_cost = merged_area_m2 * cost_per_m2

    # Create overlay for visualization
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[mask_final > 0] = (0, 255, 0)  # green overlay for canopy

    # Convert image to RGB for Streamlit
    if len(image.shape) == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(img_rgb, 1.0, overlay, ALPHA, 0)

    # === Display results ===
    st.subheader("Results with Smoothed Canopy")
    st.write(f"**Pixel Resolution (GSD):** {pixel_resolution_m:.4f} m/pixel")
    st.write(f"**Merged canopy area (pixels):** {merged_area_px:,}")
    st.write(f"**Merged canopy area (m²):** {merged_area_m2:,.2f}")
    st.write(f"**Estimated Clearing Cost:** ${estimated_cost:,.2f}")

    st.image(blended, caption="Smoothed Tree Canopy Overlay", use_container_width=True)
