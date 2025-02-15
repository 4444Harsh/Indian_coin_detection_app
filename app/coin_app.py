import streamlit as st
import cv2
import numpy as np
from collections import Counter
from PIL import Image
from ultralytics import YOLO  
model = YOLO("best (1).pt") 

def predict(image_upload):
    image = np.array(image_upload)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    result = model.predict(source=image, imgsz=640, conf=0.15)
    annotated_image = result[0].plot()
    annotated_image = annotated_image[:, :, ::-1] 
    
    detections = result[0].boxes.data
    class_names = [model.names[int(cls)] for cls in detections[:, 5]]
    count = Counter(class_names)
    
    rupee_values = {
        "1": 1,
        "2": 2,
        "5": 5,
        "10": 10
    }
    total_value = sum(rupee_values[name] for name, count in count.items())
    detection_str = ", ".join([f"{name}: {count}" for name, count in count.items()])
    
    return annotated_image, detection_str, total_value

# Streamlit UI
st.title("Indian Rupees Detection")
st.write("Upload an image to detect rupee denominations and count the total amount.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):  # Predict button
        annotated_img, detection_info, total_rupees = predict(image)
        st.image(annotated_img, caption="Annotated Image", use_column_width=True)
        st.write(f"### Detection Counts: {detection_info}")
        st.write(f"### Total Rupees: ₹{total_rupees}")
