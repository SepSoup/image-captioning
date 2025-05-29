#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset utilities for image captioning.
This module implements PyTorch dataset classes for loading and preprocessing images and captions.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from utils.vocabulary import Vocabulary

class FlickrDataset(Dataset):
    """
    PyTorch dataset class for the Flickr8k dataset.
    Loads images and their corresponding captions.
    """
    
    def __init__(self, images_dir, captions_file, vocab, transform=None, max_length=50):
        """
        Initialize the dataset.
        
        Args:
            images_dir (str): Directory containing the images
            captions_file (str): Path to the captions CSV file
            vocab (Vocabulary): Vocabulary object for text processing
            transform (torchvision.transforms, optional): Image transformations
            max_length (int): Maximum caption length
        """
        self.images_dir = images_dir
        self.df = captions_file if isinstance(captions_file, pd.DataFrame) else pd.read_csv(captions_file)
        self.vocab = vocab
        self.max_length = max_length
        
        # Define default transform if none is provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),  # Resize to 256x256
                transforms.CenterCrop(224),  # Center crop to 224x224
                transforms.ToTensor(),  # Convert to tensor (0-1)
                transforms.Normalize(  # Normalize with ImageNet mean and std
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, caption)
                image (torch.Tensor): Preprocessed image tensor
                caption (torch.Tensor): Caption token indices
        """
        #------------------------------------------------------------------------------------------------
        # TODO: Implement the data loading logic
        # 1. Get caption text and image filename from DataFrame at the given index    ✅
        # 2. Load the image from disk                                                 ✅
        # 3. Apply transformations to the image                                       ✅
        # 4. Process the caption text: convert to token indices using vocabulary      ✅
        # 5. Pad or truncate caption to max_length                                    ✅
        # 6. Convert caption to a tensor                                              ✅
        # 7. Return the processed image and caption                                   ✅

         # 1. Get caption text and image filename
        caption_text = self.df.iloc[idx]['caption']
        image_filename = self.df.iloc[idx]['image']
        image_path = os.path.join(self.images_dir, image_filename)

        # 2. Load image
        image = Image.open(image_path).convert('RGB')

        # 3. Apply transforms
        image = self.transform(image)

        # 4. Encode caption: [<start>, ..., <end>]
        tokens = self.vocab.encode(caption_text, add_special_tokens=True)

        # 5. Pad or truncate
        if len(tokens) < self.max_length:
            tokens += [self.vocab.word2idx[self.vocab.pad_token]] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]

        # 6. Convert caption to tensor
        caption = torch.tensor(tokens, dtype=torch.long)
        #------------------------------------------------------------------------------------------------
        return image, caption


class FlickrDatasetWithID(FlickrDataset):
    """
    Extended Flickr dataset that also returns image IDs.
    Useful for evaluation and visualization.
    """
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset with image ID.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, caption, image_id)
        """
        # Get base items
        image, caption = super().__getitem__(idx)
        
        # Get image ID
        img_name = self.df.iloc[idx]['image']
        
        return image, caption, img_name


def get_data_loaders(data_dir, batch_size=32, shuffle=True, num_workers=4, pin_memory=True,
                     sample_one_caption_per_image_train=True):
    """
    Create data loaders for training, validation, and testing.

    Args:
        ...
        sample_one_caption_per_image_train (bool): Whether to use one caption per image during training

    Returns:
        tuple: (train_loader, val_loader, test_loader, vocab)
    """
    max_length = 15

    # Define paths
    images_dir = os.path.join(data_dir, "processed", "images")
    train_captions = os.path.join(data_dir, "processed", "train_captions.csv")
    val_captions = os.path.join(data_dir, "processed", "val_captions.csv")
    test_captions = os.path.join(data_dir, "processed", "test_captions.csv")
    vocab_path = os.path.join(data_dir, "processed", "vocabulary.pkl")

    # Load vocabulary
    vocab = Vocabulary.load(vocab_path)

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess DataFrames
    df_train = pd.read_csv(train_captions)
    df_val = pd.read_csv(val_captions)
    df_test = pd.read_csv(test_captions)

    if sample_one_caption_per_image_train:
        df_train = df_train.groupby('image').sample(n=1, random_state=42).reset_index(drop=True)

    df_val = df_val.groupby('image').first().reset_index()
    df_test = df_test.groupby('image').first().reset_index()

    # Create datasets
    train_dataset = FlickrDataset(images_dir, df_train, vocab, transform=train_transform, max_length=max_length)
    val_dataset = FlickrDataset(images_dir, df_val, vocab, transform=val_transform, max_length=max_length)
    test_dataset = FlickrDatasetWithID(images_dir, df_test, vocab, transform=val_transform, max_length=max_length)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, vocab