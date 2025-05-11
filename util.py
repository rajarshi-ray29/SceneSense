import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import classification_report


def train_model(model, train_loader, optimizer, criterion, device, num_epochs=10, max_grad_norm=5.0):
    model.train()
    model.to(device)

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0

        for batch in train_loader:
            # Unpack batch
            text_emb = batch.get("text_emb").to(device) if "text_emb" in batch and batch["text_emb"] is not None else None
            audio_emb = batch.get("audio_emb").to(device) if "audio_emb" in batch and batch["audio_emb"] is not None else None
            vision_emb = batch.get("vision_emb").to(device) if "vision_emb" in batch and batch["vision_emb"] is not None else None
            if vision_emb is not None:
                vision_emb = vision_emb.to(device)
            else:
                # If missing, replace with zeros of correct shape
                B = text_emb.size(0) if text_emb is not None else audio_emb.size(0)
                vision_emb = torch.zeros((B, 174, 512), device=device)

            labels = batch["label"].to(device)  # shape: (B,7)
            labels = torch.argmax(labels, dim=1)

            # Forward
            logits = model(text_emb=text_emb, audio_emb=audio_emb, vision_emb=vision_emb)
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


def test_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    model.to(device)
    
    total_loss = 0.0
    preds = []
    gt = []
    
    with torch.no_grad():  # No need to compute gradients during testing
        for batch in test_loader:
            # Unpack batch
            text_emb = batch.get("text_emb").to(device) if "text_emb" in batch and batch["text_emb"] is not None else None
            audio_emb = batch.get("audio_emb").to(device) if "audio_emb" in batch and batch["audio_emb"] is not None else None
            vision_emb = batch.get("vision_emb")
            if vision_emb is not None:
                vision_emb = vision_emb.to(device)
            else:
                # If missing, replace with zeros of correct shape
                B = text_emb.size(0) if text_emb is not None else audio_emb.size(0)
                vision_emb = torch.zeros((B, 174, 512), device=device)
            
            labels = batch["label"].to(device)  # shape: (B,7)
            labels = torch.argmax(labels, dim=1)

            # Forward pass
            logits = model(text_emb=text_emb, audio_emb=audio_emb, vision_emb=vision_emb)
            loss = F.cross_entropy(logits, labels)

            # Compute the total loss
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits, dim=1)
            preds.extend(predicted.cpu().numpy())
            gt.extend(labels.cpu().numpy())

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    

    print(f"Test Loss: {avg_loss:.4f}")
    print(classification_report(np.array(gt), np.array(preds)))
