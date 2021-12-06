# Emotion_Detection-
Detection Emotion using speech data

# Overview 
This is a Classification problem using recorded audio of human speech as it's data. This project will demonstrate how to handle audio data, create synthetic data from the original audio, extract features and input them into a Convolutional Neural Network. Some of the methods used to handle the audio data are  


## Use Case 
It can sometimes be very difficult for a doctor to diagnose their patients and in the past few years we have seen many fields take advantage of what machine learning has to offer and the medical industry has also dipped it's toes in the idea. One great use case is using recorded speech data to detect emotions. This could be use full for doctors and therapist when they are evaluating their patients. A session between a doctor and their patient could be recorded and than ran through our model to predict what emotions are detected. This can be extremely use full for doctors and therapist who may miss something when evaluating a patient and can be great for seeing what subjects bring up what emotions. 


## Speech and Emotions 





![Emotions and Loudness](https://github.com/Jyve00/Emotion_Detection-/blob/main/Emotion%20Classes.png)


## Audio Data

Fast Fourier Transform 
(https://www.youtube.com/watch?v=h7apO7q16V0&ab_channel=Reducible)

![Fast Fourier Transform](https://github.com/Jyve00/Emotion_Detection-/blob/main/FFT.png)

librosa.load 
sample rate 
specrogram 


test
## Feature Extraction 
MelSpectogram
MFCC
MFCC Delta
MFCC Delta-Delta
Root Mean Square Value
Zero-crossing 
## Data Augmentation 
Butterworth Filter 
adding noise 
Pitch Stretching 
Time Stretching 
Time shifting 
Pre Emphasis 

## Modeling 

Tensorflow 
CNN 


## Evaluation 
()




## Repository Structure

```

├── Data                                <- folder containing csv data and nested subfolder of audio data
│   └── ...
├── images                              <- 
│   └── ...
├── .gitattributes                      <- file specifying files for git lfs to track
├── .gitignore                          <- file specifying files/directories to ignore
├── EDA.ipynb                           <- notebook detailing the data science process containing code and narrative
├── README.md                           <- Top-level README
├── presentation.pdf                    <- presentation slides for a business audience
└── 