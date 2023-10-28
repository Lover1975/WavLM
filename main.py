import os
import torch
from WavLM import WavLM, WavLMConfig
import torchaudio
import glob

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
batch_size = 70  # Adjusted batch size to 70

# Process files in batches
batch_counter = 1
counter = 1
for i in range(0, len(wav_file_paths), batch_size):
    print("\n")
    print(f"Batch number {batch_counter}")
    batch_counter += 1
    batch_paths = wav_file_paths[i:i + batch_size]
    batch_waveforms = []
    batch_representations = []
    batch_layer_results = []
    batch_layer_reps = []

    for wav_file_path in batch_paths:
        counter += 1
        print("\n")
        print(f"File number {counter} {wav_file_path}")
        # Load the audio file
        waveform, sample_rate = torchaudio.load(wav_file_path)
        
        # Trim or pad waveform to have exactly 60000 frames
        if waveform.shape[1] > 60000:
            waveform = waveform[:, :60000]  # Trim
        elif waveform.shape[1] < 60000:
            padding = torch.zeros((waveform.shape[0], 60000 - waveform.shape[1]))  # Pad
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
    layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
    
    # Append representations to the batch lists
    batch_representations.append(rep)
    batch_layer_results.append(layer_results)
    batch_layer_reps.append(layer_reps)
    # print(f"rep: {rep.shape}")
    # print(f"layer_results {len(layer_results)}")
    # print(f"layer_reps {len(layer_reps)}")
    # print("\n")

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