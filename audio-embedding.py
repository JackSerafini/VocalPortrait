import os
import torch
import librosa
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

def load_audio(path):
    audio, sr = librosa.load(path, sr=16000, mono=True)
    return torch.tensor(audio)

data_root = "data/mavceleb_v1_train/voices"
results = []


# Walk through all subdirectories and files
for root, dirs, files in os.walk(data_root):
    #print(dirs)
    for file in tqdm(files):
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            try:
                waveform = load_audio(file_path)
                inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state
                    audio_embedding = torch.mean(embeddings, dim=1).squeeze().numpy()  # shape: [hidden_size]
                results.append([file_path] + audio_embedding.tolist())
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


embedding_dim = len(results[0]) - 1 if results else 0
columns = ["file_path"] + [f"dim_{i}" for i in range(embedding_dim)]
df = pd.DataFrame(results, columns=columns)

df.to_csv("data/audio_embeddings.csv", index=False)
