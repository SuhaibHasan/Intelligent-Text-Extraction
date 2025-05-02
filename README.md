# Intelligent-Text-Extraction

# 🚗 Automatic Number Plate Recognition (ANPR) using Deep Learning

This project demonstrates a complete OCR pipeline for **Automatic Number Plate Recognition (ANPR)** using **Keras**, **TensorFlow**, and **Connectionist Temporal Classification (CTC)** loss. It trains a neural network to recognize license plate numbers from images.


## 🔍 Features

- Character-level license plate recognition
- Custom dataset-ready structure
- Uses CNN + Bidirectional GRU + CTC Loss
- Trained and tested on synthetic ANPR dataset
- Visual activation map of model predictions

---

## 📁 Project Structure
- anpr-ocr/
├── anpr_ocr/
│ ├── anpr_ocr__train/ # Training images and annotations
│ ├── anpr_ocr__test/ # Testing images and annotations
├── TextExtraction.ipynb # Main training and inference script
├── README.md # This file

💡 How it Works
- Images are fed into a CNN+RNN model
- Bidirectional GRUs extract sequential features
- Softmax outputs character probabilities
- CTC loss decodes text without needing explicit alignment

📌 Requirements
- Python 3.8
- TensorFlow 1.15 or 2.x (with compatibility mode)
- Keras 2.2.4
- Numpy, Matplotlib, OpenCV

🧪 Dataset
- You can use your own dataset by placing it under:
- anpr_ocr/anpr_ocr__train/img/
- anpr_ocr/anpr_ocr__train/ann/
- Each image must have a corresponding label in .json format.

🤝 Contributing
- Pull requests are welcome! For major changes, please open an issue first.

📄 License
- This project is licensed under the MIT License.
