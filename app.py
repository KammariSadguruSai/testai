import streamlit as st
from utils.ocr import ocr_and_tts
from utils.object_detection import detect_objects
from utils.scene_understanding import generate_scene_description

st.title("AI Assistive Solution for the Visually Impaired")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image_path = f"temp/{uploaded_file.name}"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Scene Understanding
    if st.button("Generate Scene Description"):
        description = generate_scene_description(image_path)
        st.write("Scene Description:")
        st.write(description)

    # OCR and TTS
    if st.button("Read Text from Image"):
        text = ocr_and_tts(image_path)
        st.write("Extracted Text:")
        st.write(text)

    # Object Detection
    if st.button("Detect Objects"):
        detected_image = detect_objects(image_path)
        st.image(detected_image, caption="Detected Objects", use_column_width=True)
