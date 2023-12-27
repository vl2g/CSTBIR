from PIL import Image
import torch
from torch import nn, optim
import glob
import os
import pandas as pd
import json
import numpy as np
from src import clip
from torch.utils.data import Dataset, DataLoader, BatchSampler
from tqdm import tqdm
import random
from dataloader import CSTBIR_dataset
import yaml
from utils import *

random.seed(110)

# Load the configuration
config_path = 'config.yaml'
config = load_config(config_path)

os.makedirs(config['training']['save_model_path'], exist_ok=True)

# # Preparing Model and Data
device = "cuda" if config['training'].config['gpu'] and torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(config['model']['model_name'], device=device, jit=False)

sketch_embedding_dict = torch.load(config['data']['sketch_embeddings_path'])

print('loading dataset...')
train_data = CSTBIR_dataset(config['data']['dataset_path'], config['data']['images_path'], config['data']['images_path'], config['data']['train_split_name'], preprocess)
val_data = CSTBIR_dataset(config['data']['dataset_path'], config['data']['images_path'], config['data']['images_path'], config['data']['val_split_name'], preprocess)
n_train_samples = len(train_data) * config['training']['batch_size']
n_val_samples = len(train_data) * config['training']['batch_size']
print('loaded dataset')
print(f'# train samples: {n_train_samples}')
print(f'# val samples: {n_val_samples}')

if device == "cpu":
    model.float()

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_data)*config['training']['epochs'])

softmax = torch.nn.Softmax(dim=1)

best_te_loss = 1e5
best_te_acc = 0.
best_ep = -1

step = 0
te_loss = 0
te_acc = 0.
with torch.no_grad():
    model.eval()
    test_pbar = tqdm(range(len(val_data)), leave=False)
    for batch in test_pbar:
        step += 1
        images, texts, sketches = val_data.get_samples()
        images = images.to(device)
        texts = texts.to(device)
        sketches = sketches.to(device)
        logits_per_image, logits_per_text, attentions = model(images, texts, sketches)
        ground_truth = torch.arange(config['model']['batch_size']).to(device)

        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        te_acc += torch.sum((torch.argmax(softmax(logits_per_text), dim=1) == ground_truth)).item()
        te_loss += total_loss.item()
        test_pbar.set_description(f"test batchCE: {total_loss.item()}", refresh=True)
    te_loss /= step
    te_acc /= n_val_samples
    print(f"te_loss {te_loss}, te_acc {te_acc*100:.2f}")

for epoch in range(config['training']['epochs']):
    print(f"running epoch {epoch}, best test loss {best_te_loss} best test acc {best_te_acc*100:.2f} after epoch {best_ep}")
    step = 0
    tr_loss = 0
    tr_acc = 0.
    model.train()
    pbar = tqdm(range(len(train_data)))
    for batch in pbar:
        step += 1
        optimizer.zero_grad()

        images, texts, sketches = train_data.get_samples()
        images = images.to(device)
        texts = texts.to(device)
        sketches = sketches.to(device)
        logits_per_image, logits_per_text, attentions = model(images, texts, sketches)
        ground_truth = torch.arange(len(images)).to(device)
        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        tr_acc += torch.sum((torch.argmax(softmax(logits_per_text), dim=1) == ground_truth)).item()
        total_loss.backward()
        tr_loss += total_loss.item()
        if device == "cpu":
            optimizer.step()
            scheduler.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            scheduler.step()
            clip.model.convert_weights(model)
        pbar.set_description(f"train batchCE: {total_loss.item()}", refresh=True)
    tr_loss /= step
    tr_acc /= n_train_samples
    
    step = 0
    te_loss = 0
    te_acc = 0.
    with torch.no_grad():
        model.eval()
        test_pbar = tqdm(range(len(val_data)), leave=False)
        for batch in test_pbar:
            step += 1
            images, texts, sketches = val_data.get_samples()
            images = images.to(device)
            texts = texts.to(device)
            sketches = sketches.to(device)
            logits_per_image, logits_per_text, attentions = model(images, texts, sketches)
            ground_truth = torch.arange(config['model']['batch_size']).to(device)

            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            te_acc += torch.sum((torch.argmax(softmax(logits_per_text), dim=1) == ground_truth)).item()
            te_loss += total_loss.item()
            test_pbar.set_description(f"test batchCE: {total_loss.item()}", refresh=True)
        te_loss /= step
        te_acc /= n_val_samples
        
    if te_loss < best_te_loss:
        best_te_loss = te_loss
        best_ep = epoch
    if te_acc > best_te_acc:
        best_te_acc = te_acc
    torch.save(model.state_dict(), os.path.join(config['training']['save_model_path'], f"model_checkpoint_{epoch}.pt"))
         
    # run["train/loss"].append(tr_loss)
    # run["train/acc"].append(te_acc * 100)
    # run["eval/loss"].append(te_loss)
    # run["eval/acc"].append(te_acc * 100)
    print(f"epoch {epoch}, tr_loss {tr_loss}, te_loss {te_loss}, tr_acc {tr_acc*100:.2f}, te_acc {te_acc*100:.2f}")
torch.save(model.state_dict(), os.path.join(config['training']['save_model_path'], "last_model.pt"))

