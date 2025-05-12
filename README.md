# Scene Sense

## Overview
This project implements various deep learning models for multimodal emotion recognition using text, audio, and visual features extracted from the MELD dataset.

## Team Members
- Omkar Nabar  
- Rajarshi Ray  
- Khandaker Abid  

## Table of Contents
- [Overview](#overview)
- [Team Members](#team-members)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models](#models)
- [Training & Evaluation](#training--evaluation)
- [Usage](#usage)

## Dataset
We use the [MELD (Multimodal EmotionLines Dataset)](https://affective-meld.github.io/) which contains conversational utterances annotated with emotions across text, audio, and visual modalities.
In the data folder, the readme.txt file contains a drive link to download the dataset.
The intial features provided by MELD and FacialMMT, are restructured and modified in format_data.py. 
The files present in the link: 
- meld_val_vision_utt.pkl
- meld_train_vision_utt.pkl
- meld_test_vision_utt.pkl
- data_emotion.p
- audio_embeddings_feature_selection_emotion.pkl

## Project Structure
```
.
├── data/
│ └── readme.txt # Link to download data
├── output/ # Output logs for different model runs
│ ├── baseline_audio_video.out
│ ├── crossattn_res_text_video.out
│ ├── hadamard_audio_video.out
│ └── ... (more log files)
├── .gitignore # Git ignore rules
├── data_loader.py # Loads and preprocesses multimodal data
├── format_data.py # Script for formatting or transforming raw data
├── main.py # Driver script to select model, load data, and train
├── models.py # Contains model definitions
├── run_model_pipeline.py # Optional automation/runner script for experiments
├── util.py # Training and evaluation utilities
├── requirements.txt # Dependencies
└── README.md # Project documentation
```

## Models
We implemented and compared the following models:
- **BaselineModel**: Concatenates modality embeddings and passes through a simple classifier.
- **CrossAttnFusionModel**: Uses cross-attention between modalities.
- **HadamardFusionModel**: Multiplies selected modality embeddings element-wise before classification.
Each model has the option to add the residual connections in the transformer encoders for text and vision. 
There is also the option to have a residual (/skip) connection in the cross attention models.

## Training and Evaluation
To train a model:
```bash
python main.py
```

## Usage
After downloading the dataset, run format_data.py (no args needed) to get the embeddings required for training+testing+validation.

In the main.py you can edit to the model you wish to train on, controlling the modalities as per the model (All models have option to have residual connection in the transformer encoder for vision and/or text. CrossAttention Model also has option to add residual connection in the Cross Attention Block.):

- Baseline : Works with unimodal, bimodal and trimodal data. 
- Hadamard Fusion: Only Bimodal.
- Cross Attention: Only Text+Vision (for now.)

### Arguments for Model:
 - use_text: Bool, use_audio: Bool, use_vision: Bool (Set True for modalities to be used)

### Arguemnts for DataLoader:
 - audio: Bool, text: Bool, vision: Bool (Set True for modalities to be used)
* For training+testing on unimodal vision data, it is advised to set the audio parameter to True alongside the vision in the Dataloader to load labels.

