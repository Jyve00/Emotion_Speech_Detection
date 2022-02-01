# custom Pytorch Data Class for the RAVDESS Dataset

import pandas as pd 
import numpy as np 
import torch 
import torchaudio
from torch.utils.data import Dataset

# Global variables 

SAMPLE_RATE = 16000
N_FFT = int(0.025 * SAMPLE_RATE)   # 25 ms 
HOP_LENGTH = int(0.01 * SAMPLE_RATE)  # 10 ms
N_MELS = 13

# Dataset Paths 

RAVDESS_dir = '/Users/stephen/Desktop/Emotion_Speech_Detection/data/RAVDESS'
RAVDESS_metadata = '/Users/stephen/Desktop/Emotion_Speech_Detection/data/metadata.csv'




# Class 

class RavdessDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 
                 target_sample_rate,
                 num_samples,
                 device, 
                 transformation=None):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)

        # if there is any transform method, apply it to audio tensor
        if self.transformation:
            signal = self.transformation(signal).to(self.device)
        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 3]


if __name__ == "__main__":
    ANNOTATIONS_FILE = "/Users/stephen/Desktop/Emotion_Speech_Detection/data/metadata.csv"
    AUDIO_DIR = "/Users/stephen/Desktop/Emotion_Speech_Detection/data/RAVDESS"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 16000*4

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    

    rav = RavdessDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device, 
                            transformation=None,)
    print(f"There are {len(rav)} samples in the dataset.")
    signal, label = rav[0]