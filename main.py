import os
import torch
from WavLM import WavLM, WavLMConfig
import torchaudio
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch import nn, optim
import numpy as np


def read_labels(file_path):
    labels_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            file_name, label = line.strip().split()
            labels_dict[file_name] = label
    return labels_dict


# Load the pre-trained checkpoints
torch.set_printoptions(threshold=torch.inf)
checkpoint = torch.load('/content/WavLM-Large.pt')
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()

# Directory where your .wav files are located (including subdirectories)
wav_directory = '/content/Voxceleb2_small'

# Use glob to find all .wav files in the directory and its subdirectories
wav_file_paths = glob.glob(os.path.join(wav_directory, '**', '*.wav'), recursive=True)

# Set batch size for processing
batch_size = 20  # Adjusted batch size to 70

# Process files in batches
batch_counter = 1
counter = 1
all_features = []
all_labels = []
batch_representations = []
batch_layer_results = []
batch_layer_reps = []
labels_dict = read_labels('/content/labels.txt')

for i in range(0, len(wav_file_paths), batch_size):
    print("\n")
    if i == 200:
      break
    print(f"Batch number {batch_counter}")
    batch_counter += 1
    batch_paths = wav_file_paths[i:i + batch_size]
    batch_waveforms = []

    for wav_file_path in batch_paths:
        if i == 200:
          break
        counter += 1
        print("\n")
        print(f"File number {counter} {wav_file_path}")
        # Load the audio file
        waveform, sample_rate = torchaudio.load(wav_file_path)
        
        # Trim or pad waveform to have exactly 60000 frames
        if waveform.shape[1] > 3000:
            waveform = waveform[:, :3000]  # Trim
        elif waveform.shape[1] < 3000:
            padding = torch.zeros((waveform.shape[0], 3000 - waveform.shape[1]))  # Pad
            waveform = torch.cat((waveform, padding), dim=1)
        
        print(f"Waveform shape: {waveform.shape}")
        # Normalize the input if needed
        if cfg.normalize:
            waveform = torch.nn.functional.layer_norm(waveform, waveform.shape)
            print(f"Waveform shape after normalization: {waveform.shape}")
        
        batch_waveforms.append(waveform)
    
    # Stack waveforms into a single tensor for batch processing
    batch_waveforms_tensor = torch.cat(batch_waveforms, dim=0)
    
    # Extract representations
    rep, layer_results = model.extract_features(batch_waveforms_tensor, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
    print(f"rep shape: {rep.shape}")
    layer_reps = [x.transpose(0, 1) for x, _ in layer_results]

    rep = rep.view(batch_size, -1)
    all_features.append(rep)
    all_labels.extend([labels_dict[path] for path in batch_paths])

    
    # Append representations to the batch lists
    batch_representations.append(rep)
    batch_layer_results.append(layer_results)
    batch_layer_reps.append(layer_reps)
    print(len(batch_representations))
    print(f"rep: {rep.shape}")
    print(f"layer_results {len(layer_results)}")
    print(f"layer_reps {len(layer_reps)}")
    print("\n")

    # Process the batch representations as needed
    # print("\n")
    # for j, wav_file_path in enumerate(batch_paths):
        # print("Representation for", wav_file_path)
        # print("Last Layer Representation:")
        # print(batch_representations[j])
        # print("Layer Results:")
        # print(batch_layer_results[j])
        # print("Layer Representations:")
        # print(batch_layer_reps[j])
# Load labels
# print("hi")
# Flatten batch_representations list and convert to tensor
print(f"all_features shape: {all_features.shape}")
print(f"Number of labels: {len(all_labels)}")
print("\n\n\n\n\n\n\n\n")
all_labels = torch.tensor(all_labels, dtype=torch.long)
print(f"Number of labels: {len(all_labels)}")
all_features = torch.cat([rep for batch in batch_representations for rep in batch], dim=0)
print(f"all_features shape: {len(all_features)}")
# print(labels_dict)
# print("hello")
# Get labels for the files you have
# print(wav_file_paths)
labels = [labels_dict[path] for path in wav_file_paths]
# print("buuu")
# Convert labels to numeric values
unique_speakers = list(set(labels))
label_to_idx = {label: idx for idx, label in enumerate(unique_speakers)}
numeric_labels = [label_to_idx[label] for label in labels]
print(f"Number of labels: {len(labels)}")
print(numeric_labels)

# Split data into training, validation, and test sets
train_features, test_features, train_labels, test_labels = train_test_split(
    all_features, numeric_labels, test_size=0.2, random_state=42
)
train_features, val_features, train_labels, val_labels = train_test_split(
    train_features, train_labels, test_size=0.25, random_state=42
)

# Define softmax classifier
class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SoftmaxClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)