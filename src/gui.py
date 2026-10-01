import cv2
import typing
import numpy as np
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from tkinter import filedialog
from mltu.configs import BaseModelConfigs


def resolve_model_configs():
    code_root = Path(__file__).resolve().parents[1]
    config_path = code_root / "config" / "configs.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    configs = BaseModelConfigs.load(str(config_path))
    if not Path(configs.model_path).is_absolute():
        configs.model_path = str((code_root / configs.model_path).resolve())

    return configs


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], model_path: str, *args, **kwargs):
        super().__init__(model_path=model_path, *args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text


def recognize_image():
    global image_path, model, result_label, label_image

    if image_path:
        image = cv2.imread(image_path)
        prediction_text = model.predict(image)
        result_label.config(text=f"Prediction: {prediction_text}")
    else:
        result_label.config(text="No image selected")
        label_image.config(image="")
        label_image.image = None


def clear_result():
    global image_path, result_label, label_image
    image_path = ""
    result_label.config(text="")
    label_image.config(image="")
    label_image.image = None


def import_image():
    global image_path
    image_path = filedialog.askopenfilename(
        initialdir="/",
        title="Select Image",
        filetypes=(("Image Files", "*.png; *.jpg; *.jpeg; *.gif"), ("All Files", "*.*")),
    )
    if image_path:
        image = Image.open(image_path)
        image.thumbnail((300, 400))
        photo = ImageTk.PhotoImage(image)
        label_image.config(image=photo)
        label_image.image = photo
        result_label.config(text="Prediction: ")
    else:
        result_label.config(text="No image selected")


if __name__ == "__main__":
    configs = resolve_model_configs()
    model = ImageToWordModel(char_list=configs.vocab, model_path=configs.model_path)

    root = tk.Tk()
    root.title("Handwriting Recognition")
    root.geometry("500x300")

    label_image = tk.Label(root)
    label_image.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

    button_import = tk.Button(
        root,
        text="Import Image",
        command=import_image,
        padx=5,
        pady=5,
        borderwidth=1,
        highlightthickness=0,
        highlightbackground="black",
    )
    button_import.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    button_recognize = tk.Button(
        root,
        text="Recognize",
        command=recognize_image,
        padx=5,
        pady=5,
        borderwidth=1,
        highlightthickness=0,
        highlightbackground="black",
    )
    button_recognize.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

    button_clear = tk.Button(
        root,
        text="Clear",
        command=clear_result,
        padx=5,
        pady=5,
        borderwidth=1,
        highlightthickness=0,
        highlightbackground="black",
    )
    button_clear.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")

    result_label = tk.Label(root)
    result_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(2, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(2, weight=1)

    root.mainloop()