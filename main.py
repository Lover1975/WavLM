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
batch_size = 1  # You can adjust this based on your available memory
batch_representations = []
batch_layer_results = []
batch_layer_reps = []
# Process files in batches
for i in range(0, len(wav_file_paths), batch_size):
    batch_paths = wav_file_paths[i:i + batch_size]
    
    for wav_file_path in batch_paths:
        print("\n")
        print(f"File number {i} {wav_file_path}")
        # Load the audio file
        waveform, sample_rate = torchaudio.load(wav_file_path)
        print(f"Waveform shape: {waveform.shape}")
        # Normalize the input if needed
        if cfg.normalize:
            waveform = torch.nn.functional.layer_norm(waveform, waveform.shape)
            print(f"Waveform shape after normalizatio: {waveform.shape}")
        # Extract representations
        rep, layer_results = model.extract_features(waveform, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        
        # Append representations to the batch lists
        batch_representations.append(rep)
        batch_layer_results.append(layer_results)
        batch_layer_reps.append(layer_reps)
        print(f"rep: {rep.shape}")
        print(f"layer_results {len(layer_results)}")
        print(f"layer_reps {len(layer_reps)}")
        print("\n")
    
    # Process the batch representations as needed
    # for j, wav_file_path in enumerate(batch_paths):
    #     print("Representation for", wav_file_path)
    #     print("Last Layer Representation:")
    #     print(batch_representations[j])
    #     print("Layer Results:")
    #     print(batch_layer_results[j])
    #     print("Layer Representations:")
    #     print(batch_layer_reps[j])

#import torch
#print(torch.__version__)
#print(torch.backends.mkldnn.is_available())
#from WavLM import WavLM, WavLMConfig

# load the pre-trained checkpoints
#checkpoint = torch.load('/mnt/4d55e72b-7ce5-4fc5-b3d4-3cfd1a4a0281/Afshari/Downloads/WavLM_Code/WavLM-Large.pt')
#cfg = WavLMConfig(checkpoint['cfg'])
#model = WavLM(cfg)
#model.load_state_dict(checkpoint['model'])
#model.eval()

# extract the representation of last layer
#wav_input_16khz = torch.randn(1,10000)
#if cfg.normalize:
#    wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)
#print(3.7)
#rep = model.extract_features(wav_input_16khz)[0]
#print(4)
# extract the representation of each layer
#wav_input_16khz = torch.randn(1,10000)
#if cfg.normalize:
#    wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)
#rep, layer_results = model.extract_features(wav_input_16khz, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
#layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
#print(rep)
#print(layer_results)
#print(layer_reps)
