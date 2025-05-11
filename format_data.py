import numpy as np
import pandas as pd
import pickle
import os, sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pickle
import os
from collections import defaultdict

# Hyperparams
max_length = 50  # Maximum length of the sentence

class UtteranceEmotionProcessor:
    
    def __init__(self):
        """
        Initialize the utterance-level data processor for emotion classification.
        Each utterance has its own label and will be processed individually.
        """
        self.max_l = max_length
        
        # Load the base dataset
        x = pickle.load(open("data/data_emotion.p", "rb"))
        revs, self.W, self.word_idx_map, self.vocab, _, self.label_index = x[0], x[1], x[2], x[3], x[4], x[5]
        self.num_classes = len(self.label_index)
        
        # Process utterances
        self.process_utterances(revs)
        
    def process_utterances(self, revs):
        """
        Process utterances and split them into train/val/test sets.
        Each utterance is treated individually with its own label.
        """
        # Organize utterances by split
        self.train_utterances = []
        self.val_utterances = []
        self.test_utterances = []
        
        # Process each utterance
        for i in range(len(revs)):
            utterance_id = f"{revs[i]['dialog']}_{revs[i]['utterance']}"
            utterance_data = {
                'id': utterance_id,
                'dialog_id': revs[i]['dialog'],
                'utterance_num': revs[i]['utterance'],
                'text': revs[i]['text'],
                'label': self.label_index[revs[i]['y']]
            }
            
            # Assign to appropriate split
            if revs[i]['split'] == "train":
                self.train_utterances.append(utterance_data)
            elif revs[i]['split'] == "val":
                self.val_utterances.append(utterance_data)
            elif revs[i]['split'] == "test":
                self.test_utterances.append(utterance_data)
        
    def get_word_indices(self, text):
        """Convert text to word indices using the vocabulary mapping"""
        words = text.split()
        length = len(words)
        indices = [self.word_idx_map.get(word, 0) for word in words]
        # Pad to max_length
        padded_indices = indices + [0] * (self.max_l - length)
        return np.array(padded_indices[:self.max_l])
    
    def get_one_hot(self, label):
        """Convert label to one-hot encoding"""
        label_arr = [0] * self.num_classes
        label_arr[label] = 1
        return np.array(label_arr)
    
    def process_text_embeddings(self):
        """
        Process text data for all utterances and save the embeddings.
        Each utterance is converted to word indices.
        """
        
        # Process each split
        train_text_data = self._process_split_text(self.train_utterances, skip_id=1169)
        val_text_data = self._process_split_text(self.val_utterances)
        test_text_data = self._process_split_text(self.test_utterances, skip_id=2052)
        
        # Save text embeddings for easy loading during training
        with open('data/utterance_text_embeddings.pkl', 'wb') as f:
            pickle.dump((train_text_data, val_text_data, test_text_data), f)
            
    
    def _process_split_text(self, utterances, skip_id = None):
        """Process text embeddings for a specific split"""
        X = []  # Features
        y = []  # Labels
        ids = []  # Utterance IDs
        i = -1
        
        for utterance in utterances:
            i+=1
            if i == skip_id:
                continue
            
            # Get word indices for the utterance text
            word_indices = self.get_word_indices(utterance['text'])
            X.append(self.W[word_indices])
            
            # Get one-hot label
            y.append(self.get_one_hot(utterance['label']))
            
            # Store utterance ID
            ids.append(utterance['id'])
        
        return {
            'features': np.array(X),
            'labels': np.array(y),
            'ids': ids
        }
    
    def process_audio_embeddings(self):
        """
        Load and process audio embeddings for all utterances.
        Audio embeddings are loaded from the existing pickle file.
        """
        
        # Load audio embeddings
        audio_path = "data/audio_embeddings_feature_selection_emotion.pkl"
        try:
            train_audio_emb, val_audio_emb, test_audio_emb = pickle.load(open(audio_path, "rb"))
            
            # Process each split
            train_audio_data = self._process_split_audio(self.train_utterances, train_audio_emb, skip_id=1169)
            val_audio_data = self._process_split_audio(self.val_utterances, val_audio_emb)
            test_audio_data = self._process_split_audio(self.test_utterances, test_audio_emb, skip_id=2052)
            
            # Save processed audio embeddings
            with open('utterance_audio_embeddings.pkl', 'wb') as f:
                pickle.dump((train_audio_data, val_audio_data, test_audio_data), f)
            
        except Exception as e:
            print(f"Error loading audio embeddings: {e}")
    
    def _process_split_audio(self, utterances, audio_emb_dict, skip_id = None):
        """Process audio embeddings for a specific split"""
        X = []  # Features
        y = []  # Labels
        ids = []  # Utterance IDs
        i = -1
        
        for utterance in utterances:
            i+=1
            if i == skip_id:
                continue
            # Get utterance ID
            utterance_id = utterance['id']
            
            # Get audio embedding if available, otherwise use zeros
            if utterance_id in audio_emb_dict:
                audio_embedding = audio_emb_dict[utterance_id]
            else:
                # Get audio embedding dimension from the first available embedding
                first_key = next(iter(audio_emb_dict))
                audio_dim = len(audio_emb_dict[first_key])
                audio_embedding = np.zeros(audio_dim)
                print(f"Missing audio for {utterance_id}")
            
            X.append(audio_embedding)
            
            # Get one-hot label
            y.append(self.get_one_hot(utterance['label']))
            
            # Store utterance ID
            ids.append(utterance_id)
        
        return {
            'features': np.array(X),
            'labels': np.array(y),
            'ids': ids
        }
    
    def process_vision_embeddings(self):
        """
        Load and process audio embeddings for all utterances.
        Audio embeddings are loaded from the existing pickle file.
        """
        
        # Load audio embeddings
        train_vision_path = "data/meld_train_vision_utt.pkl"
        val_vision_path = "data/meld_val_vision_utt.pkl"
        test_vision_path = "data/meld_test_vision_utt.pkl"
        try:
            train_vision = pickle.load(open(train_vision_path, "rb"))['train']
            train_vision_data = {'features': train_vision['vision'], 'labels': train_vision['labels']}

            val_vision = pickle.load(open(val_vision_path, "rb"))['val']
            val_vision_data = {'features': val_vision['vision'], 'labels': val_vision['labels']}

            test_vision = pickle.load(open(test_vision_path, "rb"))['test']
            test_vision_data = {'features': test_vision['vision'], 'labels': test_vision['labels']}
            
            # Save processed vision embeddings, cant put labels due to lack of space.
            with open('data/utterance_vision_embeddings.pkl', 'wb') as f:
                pickle.dump((train_vision_data, val_vision_data, test_vision_data), f)
            
        except Exception as e:
            print(f"Error loading vision embeddings: {e}")



if __name__ == "__main__":
    #vision embeddings: train set missing 1169th entry, test set missing 2052th entry
    uep = UtteranceEmotionProcessor()
    uep.process_text_embeddings()
    uep.process_audio_embeddings()
    uep.process_vision_embeddings()