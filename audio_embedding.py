import os
import torch
import librosa
import pandas as pd
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

def load_audio(path):
    audio, sr = librosa.load(path, sr=16000, mono=True)
    return torch.tensor(audio)

data_root = "data/mavceleb_train/voices"
output_dir = "data/audio_embeddings/voices"
results = []

os.makedirs(output_dir, exist_ok=True)

# Walk through all subdirectories and files
for root, dirs, files in os.walk(data_root):
    for file in tqdm(files):
        if file.endswith(".wav"):

            file_path = os.path.join(root, file)
            relative_path = file_path.split("voices/")[1]
            relative_path, _ = os.path.split(relative_path)

            try:
                waveform = load_audio(file_path)
                inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state
                    audio_embedding = torch.mean(embeddings, dim=1).squeeze().numpy()  # shape: [hidden_size]
                
                # Save embedding as .npy file
                base_filename = os.path.splitext(file)[0]
                npy_filename = f"{base_filename}.npy"
                relative_path = os.path.join(relative_path, npy_filename)

                npy_path = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                np.save(npy_path, audio_embedding)
                
                results.append([file_path] + audio_embedding.tolist())
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

embedding_dim = len(results[0]) - 1 if results else 0
columns = ["file_path"] + [f"dim_{i}" for i in range(embedding_dim)]
df = pd.DataFrame(results, columns=columns)

df.to_csv("data/audio_embeddings/audio_embeddings.csv", index=False)