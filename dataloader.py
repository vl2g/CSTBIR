import pandas as pd
import os
from PIL import Image
from src import clip
import random
import torch

class CSTBIR_dataset():
    def __init__(self, data_path, image_files_path, batch_size, split, preprocess):
        self.batch_size = batch_size
        self.data = pd.read_csv(data_path)
        self.data = self.data[self.data['split'] == split]
        self.texts = self.data['text'].to_list()
        self.categories = self.data['object_category'].to_list()
        self.images = self.data['image_filename'].to_list()
        self.images = [os.path.join(image_files_path, image_filename) for image_filename in self.images]
        self.sketches = self.data['sketch_filename'].to_list()
        self.n_samples = len(self.images)
        self.preprocess = preprocess

        self.image2sampleidx = {}
        for idx in range(len(self.images)):
            image_filename = self.images[idx]
            if image_filename not in self.image2sampleidx:
                self.image2sampleidx[image_filename] = []
            self.image2sampleidx[image_filename].append(idx)

        self.image2text = {}
        self.text2image = {}
        for idx in range(self.n_samples):
            image_filename = self.images[idx]
            text = self.texts[idx]
            if image_filename not in self.image2text:
                self.image2text[image_filename] = []
            if text not in self.text2image:
                self.text2image[text] = []
            self.image2text[image_filename].append(text)
            self.text2image[text].append(image_filename)

        self.unique_images = list(self.image2sampleidx.keys())
        assert len(self.texts) == len(self.images)
        
    def __len__(self):
        return len(self.texts) // self.batch_size

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.images[idx]))
        category = self.categories[idx]
        text = self.texts[idx]
        text = clip.tokenize(text)
        return (image, text)
    
    def is_text_conflict(self, text, images):
        image = self.text2image[text]
        if set(image) & set(images):
            return True
        else:
            return False

    def is_image_conflict(self, image, texts):
        text = self.image2text[image]
        if set(text) & set(texts):
            return True
        else:
            return False
    
    def get_samples(self):
        selected_images = []
        selected_texts = []
        processed_images = []
        processed_texts = []
        sketch_embeddings = []
        
        while len(selected_images) < self.batch_size:
            sample_idx = random.choice(range(self.n_samples))
            sample_image = self.images[sample_idx]
            sample_text = self.texts[sample_idx]
            sample_category = self.categories[sample_idx]
            sample_sketch_embedding = self.sketch_embedding_dict[self.sketches[sample_idx]]
            if sample_image in selected_images or sample_text in selected_texts:
                continue
            if self.is_text_conflict(sample_text, selected_images):
                continue
            if self.is_image_conflict(sample_image, selected_texts):
                continue
            selected_images.append(sample_image)
            selected_texts.append(sample_text)
            processed_images.append(self.preprocess(Image.open(sample_image)))
            processed_texts.append(clip.tokenize(sample_text))
            sketch_embeddings.append(sample_sketch_embedding.unsqueeze(0)) 
        
        processed_images = torch.stack(tuple(processed_images), dim=0)
        processed_texts = torch.stack(tuple(processed_texts), dim=0).squeeze(1)
        sketch_embeddings = torch.stack(tuple(sketch_embeddings), dim=0)

        return (processed_images, processed_texts, sketch_embeddings)