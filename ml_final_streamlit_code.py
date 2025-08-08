import streamlit as st
import numpy as np
from shapely.geometry import box, Polygon
from shapely.ops import unary_union

# Function to get cv2 when needed
def get_cv2():
    import cv2
    return cv2

# Example: function to load and process image
def process_image(image_path):
    cv2 = get_cv2()
    img = cv2.imread(image_path)
    # Example processing: convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

# Example: calculate area from a polygon
def calculate_area(coords):
    polygon = Polygon(coords)
    return polygon.area

# Streamlit UI
st.title("ML Final Project - Image Processing & Area Calculation")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    cv2 = get_cv2()
    # Read image from the uploaded file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, channels="BGR", caption="Uploaded Image")

    # Example: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    st.image(gray, caption="Grayscale Image", channels="GRAY")

    # Dummy coordinates for polygon (example only)
    coords = [(0,0), (0,10), (10,10), (10,0)]
    area = calculate_area(coords)
    st.write(f"Calculated Area: {area}")
