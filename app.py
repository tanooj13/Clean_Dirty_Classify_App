import streamlit as st
import pickle as pk
import numpy as np
from PIL import Image

def main():
    st.title("Clean Room, Dirty Room Classifier")

    page = st.sidebar.selectbox("Choose a page", ["Upload Image", "Classify"])

    if page == "Upload Image":
        upload_page()
    elif page == "Classify":
        classify_page()

def upload_page():
    st.write("Please upload the image of your room")

    uploaded_files = st.file_uploader("Choose JPG files", type=["jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Display the file name
            st.write("filename:", uploaded_file.name)
            # Load the image with PIL
            image = Image.open(uploaded_file)
            # Save the image for classification
            st.session_state['image'] = image
            st.session_state['filename'] = uploaded_file.name
            st.success("Image uploaded successfully. Go to 'Classify' page to see the result.")

def preprocess_image(image):
    # Convert the image to grayscale and resize to the model's expected input size
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((128, 128))  # Resize to 128x128 or whatever your model expects
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = image_array.flatten()  # Flatten the image
    return image_array

def classify_page():
    if 'image' not in st.session_state:
        st.warning("Please upload an image first.")
        return

    image = st.session_state['image']
    filename = st.session_state['filename']

    with open('clean_model', 'rb') as file:
        model = pk.load(file)

    image_array = preprocess_image(image)
    valid = model.predict([image_array])

    # Display the image
    st.image(image, caption=filename)
    if valid == 1:
        st.write("Clean Room")
    else:
        st.write("Dirty Room")

if __name__ == "__main__":
    if 'image' not in st.session_state:
        st.session_state['image'] = None
    if 'filename' not in st.session_state:
        st.session_state['filename'] = None
    main()
