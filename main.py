import models
import data_loader
import torch
import torch.nn as nn
import torch.optim as optim
from util import train_model, test_model
import os

model = models.BaselineModel(use_text=True, use_audio=True, use_vision=True)
# model = models.CrossAttnFusionModel()
# model = models.HadamardFusionModel(use_text=True, use_audio=False, use_vision=True)

# For unimodal vision keep video=True, audio=True
train_loader, val_loader, test_loader = data_loader.load_data(audio = True, text = True, vision = True)


optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_model(model, train_loader, optimizer, criterion, device, num_epochs=20)

# # Save the trained model
# model_save_path = "models"
# os.makedirs(model_save_path, exist_ok=True)
# model_filename = os.path.join(model_save_path, "cross_attn_fusion_model.pt")

# # Save the full model
# torch.save(model.state_dict(), model_filename)
# print(f"Model saved to {model_filename}")

test_model(model, test_loader, device)