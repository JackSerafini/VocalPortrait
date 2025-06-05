import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import os

def get_face(image: Image, model:YOLO):
  assert isinstance(image, Image.Image), "image must be a PIL.Image.Image instance"
  assert isinstance(model, YOLO), "model must be an instance of ultralytics.YOLO"
  
  output = model(image)
  results = Detections.from_ultralytics(output[0])
  
  if len(results.xyxy) == 0:
    raise ValueError("No faces detected in the image.")

  # Take the first detected face
  x1, y1, x2, y2 = map(int, results.xyxy[0])
  width, height = image.size
  margin = 50
  x1m = max(x1 - margin, 0)
  y1m = max(y1 - 2*margin, 0)     # to consider the hairs
  x2m = min(x2 + margin, width)
  y2m = min(y2 + margin, height)
  face_image = image.crop((x1m, y1m, x2m, y2m))
  
  return face_image

# ---- MAIN ----
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model = YOLO(model_path)

image_folder = "./video"
img_folder = "./img"
image_names = os.listdir(image_folder)

for image_name in image_names:
  try:
    starting_image = Image.open(f"{image_folder}/{image_name}")
    face_image = get_face(starting_image, model)
    output_img_name = os.path.splitext(image_name)[0]
    face_image.save(f"{img_folder}/{output_img_name}.jpg")
    print(f"Processed {image_name} successfully.")
  except Exception as e:
    print(f"Error processing {image_name}: {e}")

