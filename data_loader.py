import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class MultimodalDataset(Dataset):
    def __init__(self):
        self.audio = None
        self.vision = None
        self.text = None
        self.labels = None

    def add_audio(self, audio):
        self.audio = audio
    def add_vision(self, vision):
        self.vision = vision
    def add_text(self, text):
        self.text = text
    def add_labels(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {"label": self.labels[idx]}
        if self.audio is not None:
            sample['audio_emb'] = self.audio[idx]
        if self.vision is not None:
            sample['vision_emb'] = self.vision[idx]
        if self.text is not None:
            sample["text_emb"] = self.text[idx]
        return sample

def load_data(audio=False, vision=False, text=False, batch_size=32, data_dir="./data"):
    if not (audio or vision or text):
        print("No modality selected.")
        return None, None, None
    
    train_data = MultimodalDataset()
    val_data = MultimodalDataset()
    test_data = MultimodalDataset()


    if audio:
        with open(f"{data_dir}/utterance_audio_embeddings.pkl", "rb") as f:
            train_audio_data, val_audio_data, test_audio_data = pickle.load(f)
            train_data.add_audio(torch.tensor(train_audio_data['features'], dtype = torch.float))
            val_data.add_audio(torch.tensor(val_audio_data['features'], dtype = torch.float))
            test_data.add_audio(torch.tensor(test_audio_data['features'], dtype = torch.float))

            train_data.add_labels(torch.tensor(train_audio_data['labels'], dtype = torch.float))
            val_data.add_labels(torch.tensor(val_audio_data['labels'], dtype = torch.float))
            test_data.add_labels(torch.tensor(test_audio_data['labels'], dtype = torch.float))
    if vision:
        with open(f"{data_dir}/utterance_vision_embeddings.pkl", "rb") as f:
            train_vision_data, val_vision_data, test_vision_data = pickle.load(f)
            train_data.add_vision(torch.tensor(train_vision_data['features'], dtype = torch.float))
            val_data.add_vision(torch.tensor(val_vision_data['features'], dtype = torch.float))
            test_data.add_vision(torch.tensor(test_vision_data['features'], dtype = torch.float))

    if text:
        with open(f"{data_dir}/utterance_text_embeddings.pkl", "rb") as f:
            train_text_data, val_text_data, test_text_data = pickle.load(f)
            train_data.add_text(torch.tensor(train_text_data['features'], dtype = torch.float))
            val_data.add_text(torch.tensor(val_text_data['features'], dtype = torch.float))
            test_data.add_text(torch.tensor(test_text_data['features'], dtype = torch.float))

            if not audio :
                train_data.add_labels(torch.tensor(train_text_data['labels'], dtype = torch.float))
                val_data.add_labels(torch.tensor(val_text_data['labels'], dtype = torch.float))
                test_data.add_labels(torch.tensor(test_text_data['labels'], dtype = torch.float))


    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, val_loader, test_loader