import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model(
    r"F:\Python\Classification(CatVsDog)\cat_dog_cnn.h5"
)

input_h = model.input_shape[1]
input_w = model.input_shape[2]

# ======================
# TKINTER WINDOW
# ======================
root = tk.Tk()
root.geometry("800x600")
root.title("Cat Vs Dog Classifier")
root.config(bg="white")

original_image = None
canvas_image = None

# ======================
# FUNCTIONS
# ======================
def add_image():
    global original_image, canvas_image

    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    original_image = Image.open(file_path).convert("RGB")

    display_img = original_image.resize((400, 400))
    canvas_image = ImageTk.PhotoImage(display_img)

    canvas.delete("all")
    canvas.config(width=400, height=400)
    canvas.create_image(0, 0, image=canvas_image, anchor="nw")

    result_label.config(text="")

def predict_image():
    if original_image is None:
        result_label.config(text="Please add an image first!", fg="red")
        return

    img = original_image.resize((input_w, input_h))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0][0]

    if prediction > 0.5:
        result_label.config(text="It is a Dog 🐶", fg="green")
    else:
        result_label.config(text="It is a Cat 🐱", fg="blue")

def clear_canvas():
    global original_image, canvas_image
    canvas.delete("all")
    original_image = None
    canvas_image = None
    result_label.config(text="")

# ======================
# UI
# ======================
left_frame = tk.Frame(root, width=200, height=400, bg="white")
left_frame.pack(side="left", fill="y")

canvas = tk.Canvas(root, width=400, height=400, bg="lightgray")
canvas.pack(pady=20)

result_label = tk.Label(
    root, text="", font=("Arial", 18, "bold"), bg="white"
)
result_label.pack(pady=10)

tk.Button(left_frame, text="Add Image", command=add_image).pack(pady=20)
tk.Button(left_frame, text="Predict", command=predict_image).pack(pady=10)
tk.Button(left_frame, text="Clear", command=clear_canvas, bg="#FF9798").pack(pady=10)

root.mainloop()
