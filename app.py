import streamlit as st
from PIL import Image
import cv2
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os

# Memuat model yang telah dilatih
model_path = 'modela85va89.h5'
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
            'putih': 'Merah Cerah, Cokelat Gelap, Pirang Tajam, Hitam, Navy',
            'kuning': 'Biru Muda, Merah Muda Pastel, Hijau Mint, Lavender, Peach',
            'coklat': 'Merah Hati, Kuning Mustard, Hijau Zaitun, Coral, Cokelat Muda',
            'hitam': 'Putih Tulang, Emas, Perak, Biru Kobalt, Magenta'
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
    st.write("color outfit recommendation: ", recommended_outfit)

    # Menampilkan palet warna berdasarkan hasil prediksi
    color_palette_paths = {
    'putih': ['pallete1.jpg', 'palleteputih2.jpg', 'palleteputih3.jpg', 'palleteputih4.jpg', 'palleteputih5.jpg', 'palleteputih6.jpg'],
    'kuning': ['pallete1.jpg', 'palleteputih2.jpg', 'palleteputih3.jpg', 'palleteputih4.jpg', 'palleteputih5.jpg', 'palleteputih6.jpg'],
    'coklat': ['pallete1.jpg', 'palleteputih2.jpg', 'palleteputih3.jpg', 'palleteputih4.jpg', 'palleteputih5.jpg', 'palleteputih6.jpg'],
    'hitam': ['pallete1.jpg', 'palleteputih2.jpg', 'palleteputih3.jpg', 'palleteputih4.jpg', 'palleteputih5.jpg', 'palleteputih6.jpg']
    }

    palette_paths = color_palette_paths.get(prediction)
    
    if palette_paths:
        # Muat semua gambar palet warna
        palette_images = [Image.open(path) for path in palette_paths if os.path.exists(path)]
        
        # Tampilkan gambar palet warna sesuai tata letak yang diinginkan
        col1, col2, col3 = st.columns(3)
        col2.image(palette_images[0], width=300, caption='Palette')  # Photo 1
        col1, col2, col3 = st.columns(3)
        col1.image(palette_images[1], width=300, caption='Palette 1')  # Photo 2
        col3.image(palette_images[2], width=300, caption='Palette 2')  # Photo 3
        col1.image(palette_images[3], width=300, caption='Palette 3')  # Photo 4
        col3.image(palette_images[4], width=300, caption='Palette 4')  # Photo 5
        col1, col2, col3 = st.columns(3)
        col2.image(palette_images[5], width=300, caption='Palette 5')  # Photo 6
    else:
        st.write("Color palettes not available.")
