"""
Team Members:
- Omkar Nabar
- Rajarshi Ray
- Khandaker Abid

General Description:
This code implements a multi-modal neural network model using PyTorch to process and classify data from multiple modalities: text, audio, and vision.
It consists of individual modality encoders for each data type (TextEncoder, AudioEncoder, VisionEncoder) and several fusion models like BaselineModel,
CrossAttnFusionModel, and HadamardFusionModel that combine the features from these encoders to make predictions. These models are based on Transformer-based 
architectures and attention mechanisms for cross-modal learning. The output is passed through a classifier to generate the final classification logits.

NLP Class Concepts Used:
- The TextEncoder, VisionEncoder, and AudioEncoder utilize transformer-based architectures (which inherently follow syntactic principles) to process sequences
  and learn from them. The final classification layer in the models applies classification principles on the learned features.

- The use of multi-modal encoders (TextEncoder, AudioEncoder, VisionEncoder) implicitly captures semantic features of the inputs. The fusion models (BaselineModel, 
  CrossAttnFusionModel, HadamardFusionModel) probabilistically combine these features for classification. Cross-modal attention mechanisms and feature fusion also 
  represent probabilistic interactions between modalities.

- The TextEncoder utilizes a Transformer architecture, which is a key technique in modern language modeling. It processes textual input sequentially and captures
  complex dependencies between words (syntax and semantics).

- The various models applied in the code are customized for multi-modal classification tasks, making them suitable for applications such as image-text, text-audio, 
  and multi-modal data fusion. The HadamardFusionModel and CrossAttnFusionModel are specific examples of custom statistical fusion techniques to process multi-modal data.

System Information:
Code executed on an NVIDIA Tesla V100 GPU with Ubuntu 18.04 operating system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
 
from sklearn.metrics import classification_report
 
 
################################################################################################################################
###### INDIVIDUAL MODALITY ENCODERS (CAN return sequence./pooled output and can utilized skip connections )#####################
class TextEncoder(nn.Module):
    def __init__(self, input_dim=300, output_dim = 512, num_layers = 3, nhead=6, dim_feedforward=600, dropout=0.1, GAP = True, residual = False):
        super().__init__()
        self.GAP = GAP
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True),
            num_layers=3
        )
 
        self.lin = nn.Linear(input_dim, output_dim)
 
        self.use_residual = residual
 
        # If output_dim != input_dim, project input to output_dim for residual addition
        if input_dim != output_dim:
            self.res_proj = nn.Linear(input_dim, output_dim)
        else:
            self.res_proj = nn.Identity()
 
        self.norm = nn.LayerNorm(output_dim)
 
    def forward(self, x):  # x: (B, seq_len=50, 300)
        residual = x
        x = self.encoder(x)  # (B, 50, 300)
        if self.GAP:
            x = x.mean(dim = 1)
            x = self.lin(x)
 
 
            if self.use_residual:
                residual = residual.mean(dim=1)  # GAP on residual too
                residual = self.res_proj(residual)  # match dimensions
                x = x + residual
                x = self.norm(x)
        return x  # (B, 300)
    
 
class VisionEncoder(nn.Module):
    def __init__(self, input_dim=512, num_layers = 3, nhead=8, dim_feedforward=512, dropout=0.1, output_dim = 512, GAP = True, residual = False):
        super().__init__()
        self.GAP = GAP
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True),
            num_layers= num_layers
        )
 
        self.lin = nn.Linear(input_dim, output_dim)
 
        self.use_residual = residual
 
        # If output_dim != input_dim, project input to output_dim for residual addition
        if input_dim != output_dim:
            self.res_proj = nn.Linear(input_dim, output_dim)
        else:
            self.res_proj = nn.Identity()
 
        self.norm = nn.LayerNorm(output_dim)
 
    def forward(self, x):  # x: (B, 174, 512)
        residual = x
        x = self.encoder(x)  # (B, 174, 512)
        if self.GAP:
            x = x.mean(dim = 1)
            x = self.lin(x)
 
 
            if self.use_residual:
                residual = residual.mean(dim=1)  # GAP on residual too
                residual = self.res_proj(residual)  # match dimensions
                x = x + residual
                x = self.norm(x)
        
        return x  # (B, 300)  # (B, 512)
    
 
class AudioEncoder(nn.Module):
    def __init__(self, input_dim=1611, output_dim=512):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
 
    def forward(self, x):  # x: (B, 1611)
        return self.proj(x)  # (B, output_dim)
 
 
################################################################################################################################
####################################### BASELINE MODEL (WORKS WITH ANY LEVEL OF MODALITY) ######################################
class BaselineModel(nn.Module):
    def __init__(self, 
                 use_text=True, 
                 use_audio=True, 
                 use_vision=True,
                 modality_dim = 512, 
                 model_hidden_dim=256, 
                 num_classes=7, 
                 num_layers = 3, 
                 nhead=4, 
                 dim_feedforward=512,
                 encoder_dropout=0.1,
                 modality_dropout = 0.5,
                 t_residual = True,
                 v_residual = True):
        super().__init__()
 
        self.use_text = use_text
        self.use_audio = use_audio
        self.use_vision = use_vision
        self.modality_dropout = modality_dropout
 
        # Initialize encoders
 
        #constants
        self.TEXT_DIM = 300 # max len = 50
        self.AUDIO_DIM = 1611
        self.VISION_DIM = 512 # num frames 174
 
 
        if use_text:
            self.text_encoder = TextEncoder(input_dim= self.TEXT_DIM, output_dim = modality_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=encoder_dropout, residual=t_residual)
        if use_audio:
            self.audio_encoder = AudioEncoder(input_dim=self.AUDIO_DIM, output_dim = modality_dim)
        if use_vision:
            self.vision_encoder = VisionEncoder(input_dim=self.VISION_DIM, output_dim = modality_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=encoder_dropout, residual=v_residual)
 
 
        # Compute fused dimension
        fused_dim = 0
        if use_text:
            fused_dim += modality_dim
        if use_audio:
            fused_dim += modality_dim
        if use_vision:
            fused_dim += modality_dim
 
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, model_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(model_hidden_dim, num_classes)
        )
 
    def forward(self, text_emb=None, audio_emb=None, vision_emb=None):
        features = []
 
        if self.use_text:
            pooled_text = self.text_encoder(text_emb)  # (B, 300)
            if self.training and torch.rand(1).item() < self.modality_dropout:
                pooled_text = torch.zeros_like(pooled_text)
            features.append(pooled_text)
 
        if self.use_audio:
            pooled_audio = self.audio_encoder(audio_emb)  # (B, 1611)
            if self.training and torch.rand(1).item() < self.modality_dropout:
                pooled_audio = torch.zeros_like(pooled_audio)
            features.append(pooled_audio)
 
        if self.use_vision:
            pooled_vision = self.vision_encoder(vision_emb)  # (B, 512)
            if self.training and torch.rand(1).item() < self.modality_dropout:
                pooled_vision = torch.zeros_like(pooled_vision)
            features.append(pooled_vision)
 
        fused = torch.cat(features, dim=1)
        logits = self.classifier(fused)
        return logits
    
################################################################################################################################
#######################################  CROSS MODAL ATTENTION AS FUSION (ONLY WORKS WITH TEXT AND VISION DATA AS ONLY THOSE ARE SEQUENTIAL) #######################################################
 
class CrossModalAttention(nn.Module):
    def __init__(self, text_dim, vision_dim, shared_dim=256, nhead=4, residual = False):
        super().__init__()
        # Projection to shared dimension
        self.text_proj = nn.Linear(text_dim, shared_dim)
        self.vision_proj = nn.Linear(vision_dim, shared_dim)
 
        # Attention blocks
        self.text_to_vision_attn = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=nhead, batch_first=True)
        self.vision_to_text_attn = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=nhead, batch_first=True)
 
 
        self.residual = residual
        # Norms for residual outputs
        self.text_norm = nn.LayerNorm(shared_dim)
        self.vision_norm = nn.LayerNorm(shared_dim)
 
    def forward(self, text_feats, vision_feats):
        # Project to shared dimension
        text_proj = self.text_proj(text_feats)     # (B, T_text, shared_dim)
        vision_proj = self.vision_proj(vision_feats)  # (B, T_vision, shared_dim)
 
        # Text attends to vision
        text_attn_output, _ = self.text_to_vision_attn(text_proj, vision_proj, vision_proj)  # (B, T_text, shared_dim)
        if self.residual:
            text_attn_output = self.text_norm(text_attn_output + text_proj)
        pooled_text_attn = text_attn_output.mean(dim=1)  # (B, shared_dim)
 
        # Vision attends to text
        vision_attn_output, _ = self.vision_to_text_attn(vision_proj, text_proj, text_proj)  # (B, T_vision, shared_dim)
        if self.residual:
            vision_attn_output = self.vision_norm(vision_attn_output + text_proj)
        pooled_vision_attn = vision_attn_output.mean(dim=1)  # (B, shared_dim)
 
        return pooled_text_attn, pooled_vision_attn
    
class CrossAttnFusionModel(nn.Module):
    def __init__(self, 
                 modality_dim = 512, 
                 model_hidden_dim=256, 
                 num_classes=7, 
                 t_num_layers = 3, 
                 t_nhead=6, 
                 t_dim_feedforward=512,
                 t_encoder_dropout=0.1,
                 v_num_layers = 3, 
                 v_nhead=8, 
                 v_dim_feedforward=1024,
                 v_encoder_dropout=0.1,
                 modality_dropout = 0.5,
                 attn_nhead = 8,
                 t_residual = False,
                 v_residual = False,
                 cross_attn_residual = False):
        super().__init__()
 
        #constants
        self.TEXT_DIM = 300 # max len = 50
        self.VISION_DIM = 512 # num frames 174
 
 
        self.text_encoder = TextEncoder(input_dim= self.TEXT_DIM, output_dim = modality_dim, num_layers = t_num_layers, nhead=t_nhead, dim_feedforward=t_dim_feedforward, dropout=t_encoder_dropout, GAP = False, residual=t_residual)
        self.vision_encoder = VisionEncoder(input_dim=self.VISION_DIM, output_dim = modality_dim, num_layers = v_num_layers, nhead=v_nhead, dim_feedforward=v_dim_feedforward, dropout=v_encoder_dropout, GAP = False, residual=v_residual)
 
        self.cross_attn_tv = CrossModalAttention(text_dim=self.TEXT_DIM, vision_dim=self.VISION_DIM, shared_dim=modality_dim, nhead=attn_nhead, residual=cross_attn_residual)  
 
        # Classifier
 
        self.classifier = nn.Sequential(
            nn.Linear(2*modality_dim, model_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(model_hidden_dim, num_classes)
        )
 
    def forward(self, text_emb=None, audio_emb = None, vision_emb=None):
        pooled_text_attn, pooled_vision_attn = self.cross_attn_tv(text_emb, vision_emb)
 
        features = torch.cat([pooled_text_attn, pooled_vision_attn], dim = 1)
 
        logits = self.classifier(features)
        return logits
 
################################################################################################################################
#######################################  Hadamard Fusion  ######################################################################
 
class HadamardFusionModel(nn.Module):
    def __init__(self, 
                 use_text=True, 
                 use_audio=True, 
                 use_vision=True,
                 modality_dim = 512, 
                 model_hidden_dim=256, 
                 num_classes=7, 
                 num_layers = 3, 
                 nhead=4, 
                 dim_feedforward=512,
                 encoder_dropout=0.1,
                 modality_dropout = 0.5,
                 t_residual = True,
                 v_residual = True):
        super().__init__()
        self.use_text = use_text
        self.use_audio = use_audio
        self.use_vision = use_vision
        self.modality_dropout = modality_dropout
 
        # Initialize encoders
 
        #constants
        self.TEXT_DIM = 300 # max len = 50
        self.AUDIO_DIM = 1611
        self.VISION_DIM = 512 # num frames 174
 
 
        if use_text:
            self.text_encoder = TextEncoder(input_dim= self.TEXT_DIM, output_dim = modality_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=encoder_dropout, residual=t_residual)
        if use_audio:
            self.audio_encoder = AudioEncoder(input_dim=self.AUDIO_DIM, output_dim = modality_dim)
        if use_vision:
            self.vision_encoder = VisionEncoder(input_dim=self.VISION_DIM, output_dim = modality_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=encoder_dropout, residual=v_residual)
 
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(modality_dim, model_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(model_hidden_dim, num_classes)
        )
 
    def forward(self, text_emb=None, audio_emb=None, vision_emb=None):
        features = []
 
        if self.use_text and self.use_audio:
            pooled_text = self.text_encoder(text_emb)  # (B, 300)
            pooled_audio = self.audio_encoder(audio_emb) # (B, 1611)
            features = pooled_text * pooled_audio
 
        if self.use_audio and self.use_vision:
            pooled_audio = self.audio_encoder(audio_emb)  
            pooled_vision = self.vision_encoder(vision_emb)
            features = pooled_vision * pooled_audio
 
        if self.use_vision and self.use_text:
            pooled_vision = self.vision_encoder(vision_emb)  # (B, 512)
            pooled_text = self.text_encoder(text_emb) 
            features = pooled_vision * pooled_text
 
        logits = self.classifier(features)
        return logits
    