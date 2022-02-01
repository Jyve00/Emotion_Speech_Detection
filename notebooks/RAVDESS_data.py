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

