# 🐾 Cat vs Dog Classifier

A deep learning image classifier that predicts whether an uploaded image contains a **cat** or a **dog**, built with TensorFlow and available in two interfaces: a desktop Tkinter GUI and a browser-based Streamlit web app.

---

## 📁 Project Structure

```
Classification-Cat-Vs-Dog/
├── app.py              # Streamlit web app
├── main.py             # Tkinter desktop app
├── cat_dog_cnn.h5      # Trained CNN model
└── README.md
```

---

## 🧠 Model

The classifier uses a **Convolutional Neural Network (CNN)** trained on cat and dog images, saved as `cat_dog_cnn.h5`. The model outputs a single sigmoid value:

- Value **> 0.5** → 🐶 Dog
- Value **≤ 0.5** → 🐱 Cat

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9 – 3.13**
- pip

### Installation

```bash
pip install streamlit tensorflow pillow numpy
```

---

## 🌐 Streamlit Web App (`app.py`)

A browser-based interface with image upload, prediction, and confidence metrics.

### Run

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### Features

- 📤 Upload JPG or PNG images
- 🔍 One-click prediction
- 📊 Confidence score display for both Cat and Dog
- ⚡ Model cached for fast repeated predictions

---

## 🖥️ Tkinter Desktop App (`main.py`)

A native Windows desktop GUI with buttons for adding, predicting, and clearing images.

### Run

```bash
python main.py
```

> **Note:** Update the model path in `main.py` to match your local file location:
> ```python
> model = tf.keras.models.load_model(r"path\to\cat_dog_cnn.h5")
> ```

### Features

- 📂 File dialog to browse and load images
- 🖼️ Live image preview on canvas
- 🐾 Prediction result shown with color-coded label
- 🗑️ Clear button to reset the canvas

---

## 📊 How It Works

```
Input Image
    ↓
Resize to model input size
    ↓
Normalize pixel values (÷ 255)
    ↓
CNN Model Prediction
    ↓
Sigmoid Output (0.0 – 1.0)
    ↓
< 0.5 → Cat 🐱  |  > 0.5 → Dog 🐶
```

---

## ⚠️ Troubleshooting

| Problem | Fix |
|---|---|
| `No file or directory: cat_dog_cnn.h5` | Make sure `cat_dog_cnn.h5` is in the same folder as `app.py` |
| `No matching distribution found for tensorflow` | Use Python 3.9–3.13; avoid 3.14+ |
| Streamlit shows blank page | Save `app.py` before running; check terminal for errors |
| Tkinter app doesn't open | Run `python main.py` from the correct directory |

---

## 🛠️ Built With

- [TensorFlow / Keras](https://www.tensorflow.org/) — model training & inference
- [Streamlit](https://streamlit.io/) — web interface
- [Tkinter](https://docs.python.org/3/library/tkinter.html) — desktop GUI
- [Pillow](https://pillow.readthedocs.io/) — image processing
- [NumPy](https://numpy.org/) — array operations

---

## 📄 License

This project is for educational purposes.
