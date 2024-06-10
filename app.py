import streamlit as st
from PIL import Image
import cv2
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# Memuat model yang telah dilatih
model_path ='model3.h5'
model = load_model(model_path)

# Daftar kelas
classes = ['putih', 'kuning', 'coklat', 'hitam']

# Fungsi untuk memproses gambar
def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Deteksi wajah
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(image_rgb)

    if faces:
        # Ambil wajah pertama yang terdeteksi
        face = faces[0]
        x, y, width, height = face['box']
        # Memotong gambar sesuai bounding box
        cropped_face = image[y:y+height, x:x+width]

        # Proses prediksi skin tone
        face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_array = img_to_array(face_resized)
        face_array = np.expand_dims(face_array, axis=0)
        face_array = tf.keras.applications.mobilenet_v3.preprocess_input(face_array)

        # Prediksi kelas skin tone
        prediction = model.predict(face_array)
        predicted_class = classes[np.argmax(prediction)]

        # Menambahkan rekomendasi outfit berdasarkan warna kulit
        outfit_recommendations = {
            'putih': 'Warna-warna gelap seperti hitam, biru dongker, atau merah marun.',
            'kuning': 'Warna-warna cerah seperti putih, krem, atau pastel.',
            'coklat': 'Warna-warna netral seperti abu-abu, coklat muda, atau biru tua.',
            'hitam': 'Warna-warna terang seperti putih, kuning, atau merah.'
        }
        recommended_outfit = outfit_recommendations.get(predicted_class, 'No recommendation available.')

        return predicted_class, recommended_outfit
    else:
        return "No face detected", "No recommendation available"

# Membuat aplikasi Streamlit
st.title("Skin Tone and Outfit Recommendation App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Processing...")

    # Save the uploaded image to a temporary file and process it
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    prediction, recommended_outfit = process_image("temp.jpg")

    st.write("Prediction: ", prediction)
    st.write("Recommended Outfit Colors: ", recommended_outfit)
