import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import h5py

# User authentication
users = {"admin": "admin123", "user": "user123"}

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""


def signup():
    st.subheader("Sign Up")
    new_username = st.text_input("Choose a username")
    new_password = st.text_input("Choose a password", type="password")
    if st.button("Sign Up"):
        if new_username in users:
            st.warning("Username already exists. Try a different one.")
        else:
            users[new_username] = new_password
            st.success("Account created successfully! Please log in.")


def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.success(f"Welcome {username}!")
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")


def logout():
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""
    st.rerun()


if not st.session_state["authenticated"]:
    option = st.sidebar.radio("Authentication", ["Login", "Sign Up"])
    if option == "Login":
        login()
    else:
        signup()
    st.stop()

st.sidebar.button("Logout", on_click=logout)

# Load model
img_size = (512, 512)
n_classes = 5  # Change this based on the number of classes in your dataset

# Recreate model architecture
base_model = EfficientNetB3(
    include_top=False,
    weights=None,
    input_shape=(img_size[0], img_size[1], 3),
    pooling="max",
)
base_model.trainable = True
x = base_model.output
x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.999, epsilon=0.001)(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

# Load trained weights
h5py.File("final_model_weights.weights.h5", "r")

# Define class labels (Modify this according to your classes)
class_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# Streamlit UI
st.title("Diabetic Retinopathy Classification using EfficientNetB3")
st.write(f"Logged in as: {st.session_state['username']}")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_pil = image_pil.resize(img_size)
    img_array = image.img_to_array(image_pil)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    # Display image and result
    st.image(image_pil, caption="Uploaded Image", use_container_width=True)
    st.write(f"**Prediction:** {class_labels[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}%")
