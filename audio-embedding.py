from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import librosa

# Load model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")


def load_audio(path):
    audio, sr = librosa.load(path, sr=16000, mono=True)
    return torch.tensor(audio)

audio_path = "data/audio/1.mp3"
waveform = load_audio(audio_path)

# Preprocess the waveform
inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")

# Extract embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # shape: [batch_size, time_steps, hidden_size]

# Mean pooling over time
audio_embedding = torch.mean(embeddings, dim=1)  # shape: [batch_size, hidden_size]
