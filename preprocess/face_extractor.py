import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import os
from tqdm import tqdm

def get_face(image: Image, model:YOLO):
  assert isinstance(image, Image.Image), "image must be a PIL.Image.Image instance"
  assert isinstance(model, YOLO), "model must be an instance of ultralytics.YOLO"
  
  output = model.predict(image, verbose=False)
  results = Detections.from_ultralytics(output[0])
  
  if len(results.xyxy) == 0:
    raise ValueError("No faces detected in the image.")

  # Take the first detected face
  x1, y1, x2, y2 = map(int, results.xyxy[0])
  width, height = image.size
  margin = 10
  x1m = max(x1 - margin, 0)
  y1m = max(y1 - 2*margin, 0)     # to consider the hairs
  x2m = min(x2 + margin, width)
  y2m = min(y2 + margin, height)
  face_image = image.crop((x1m, y1m, x2m, y2m))
  
  return face_image

# ---- MAIN ----
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model = YOLO(model_path)
data_root = "data/mavceleb_v1_train/faces"
save_path = "data/mavceleb_v1_train_cropped/faces"

for root, dirs, files in os.walk(data_root):
    for file in tqdm(files):
        if file.endswith(".jpg"):
            file_path = os.path.join(root, file)
            relative_path = file_path.split("faces/")[1]

            try:
                starting_image = Image.open(file_path)
                face_image = get_face(starting_image, model)

                save_full_path = os.path.join(save_path, relative_path)
                os.makedirs(os.path.dirname(save_full_path), exist_ok=True)

                face_image.save(save_full_path)
            except Exception as e:
                print(f"Error processing {relative_path}: {e}")


