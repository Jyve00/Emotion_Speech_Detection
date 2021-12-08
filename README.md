# Emotion_Detection-
Detection Emotion using speech data
## Overview 
 There's a lot that the human voice can tell us beyond just understanding what someone is saying but also how they are feeling and what emotions they are displaying. It can sometimes be obvious that an individual is happy or sad just by the tonal qualities of their voice and how they speak. Computers can also detect these differences and qualities in tone over the different emotions but it is a very hard problem to solve due to the differences in the human voice across the globe. But there are so many benefits that analyzing human speech can bring and there are many different ways to go about this problem. This project will be my approach to dealing with human speech data and using that data to come up with a solution to an already existing problem. This project will also showcase my skills as a Data Scientist as well as a professional Audio Engineer. 

# Business Summary 

 It Can sometimes be difficult for doctors and therapists to pickup on emotions just because of the complexity of humans in general but a lot of the time you can tell how an individual is feeling just by the tone of their voice, so I am purposing a system that records a session between a therapist and a patient and it can detect an emotion that the therapist might have missed . It Can create a clearer picture of what subjects bring up what emotions and can keep a log of these emotions and can help diagnose patients. This system could also be implemented in a smart watch application that could monitor the patient at all times and notify the user when they are expressing a certain emotion. For example it could notify someone with anger management issues that they are losing control of there emotions and it could help the user calm down. 




# Data Understanding 

Source: https://smartlaboratory.org/ravdess/

The RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) is a widely used dataset for emotion classification using recorded speech because of its high quality and consistent audio quality. The dataset contains 24 actors (12 female and 12 male) vocalizing 2 different lexically-matched statements in a neutral North American accent. Each statement is spoken in 8 emotion classes with each class performed in 2 intensities (normal and intense). Each audio file is in .wav format, mono (1 channel) and with a sampling rate of 48,000 Hz. The duration of each track is between 3 and 5 seconds. 


Academic paper citation from the creators of RAVDESS:  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391




![Emotions and Loudness](https://github.com/Jyve00/Emotion_Detection-/blob/main/Emotion%20Classes.png)


## Modeling 
 
For this problem I chose to stick with using Convolution Neural Networks and test with various architectures and hyperparameters. 


## Evaluation 
(




## Repository Structure

```

├── Data                                <- contains csv data and nested subfolder of audio data
│   └── ...
├── images                             <- contains saved images from notebooks. 
│   └── ...
├── notebooks                          <- contains more in depth notebooks 
│  └── EDA.ipynb                       <- contains detailing the data science process containing code and narrative
│  └── first_model.ipynb                <- contains basic data preprocessing and NN model
│  └── MVP.ipynb
│  └──              <- notebook containing Minimum Viable Product as a simple feature extraction and CNN
├── .gitattributes                      <- file specifying files for git lfs to track
├── .gitignore                          <- file specifying files/directories to ignore
├── Emotion_Classifier.ipynb            <- Main Notebook 
├── README.md                           <- Top-level README
├── presentation.pdf                    <- presentation slides for a business audience
└── 