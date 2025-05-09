#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CNN Encoder module for image captioning.
This module implements the encoder part of the image captioning system.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    """
    CNN Encoder for extracting feature representations from images.
    Uses a pre-trained CNN backbone with the classification head removed.
    """
    
    def __init__(self, model_name='inception_v3', embed_size=256, pretrained=True, trainable=False):
        """
        Initialize the encoder.
        
        Args:
            model_name (str): Name of the CNN backbone to use
                Supported models: 'resnet18', 'resnet50', 'mobilenet_v2', 'inception_v3'
            embed_size (int): Dimensionality of the output embeddings
            pretrained (bool): Whether to use pre-trained weights
            trainable (bool): Whether to fine-tune the CNN backbone
        """
        super(EncoderCNN, self).__init__()
        
        self.model_name = model_name.lower()
        self.embed_size = embed_size
        
        import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    """
    CNN Encoder for extracting feature representations from images.
    Uses a pre-trained CNN backbone with the classification head removed.
    """
    
    def __init__(self, model_name='inception_v3', embed_size=256, pretrained=True, trainable=False):
        """
        Initialize the encoder.
        
        Args:
            model_name (str): Name of the CNN backbone to use ('resnet18', 'resnet50', 'mobilenet_v2', 'inception_v3')
            embed_size (int): Dimensionality of the output embeddings
            pretrained (bool): Whether to use pre-trained weights
            trainable (bool): Whether to fine-tune the CNN backbone
        """
        super(EncoderCNN, self).__init__()
        
        self.model_name = model_name.lower()
        self.embed_size = embed_size
        
        #------------------------------------------------------------------------------------------------
        # BackBone Selection
        match model_name:
            case 'resnet18' :
                self.cnn = models.resnet18(pretrained=pretrained)
                in_features = self.cnn.fc.in_features
                self.cnn.fc = nn.Identity()
        
            case 'resnet50' :
                self.cnn = models.resnet50(pretrained=pretrained)
                in_features = self.cnn.fc.in_features
                self.cnn.fc = nn.Identity()
        
            case 'mobilenet_v2' :
                self.cnn = models.mobilenet_v2(pretrained=pretrained)
                in_features = self.cnn.classifier[1].in_features
                self.cnn.classifier = nn.Identity()
        
            case 'inception_v3' :
                self.cnn = models.inception_v3(pretrained=pretrained, aux_logits=False)
                in_features = self.cnn.fc.in_features
                self.cnn.fc = nn.Identity()

        # Projection head 
        # Linear → BatchNorm → ReLU → Dropout
        # (feature dimension -> embed_size)
        self.project = nn.Sequential(
            nn.Linear(in_features, embed_size),
            nn.BatchNorm1d(embed_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # freeze CNN
        if not trainable:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # TO DO: Initialize and configure the CNN backbone based on model_name

        # 1. Create the CNN model using torchvision.models with pretrained weights if specified ✅
        # 2. Store the feature dimension size (before the final classifier) ✅
        # 3. Remove the classifier/fully-connected layer and replace with nn.Identity() ✅
        #------------------------------------------------------------------------------------------------


      
    
    def forward(self, images):
        """
        Forward pass to extract features from images.
        
        Args:
            images (torch.Tensor): Batch of input images [batch_size, 3, height, width]
            
        Returns:
            torch.Tensor: Image features [batch_size, embed_size]
        """
        # Extract features from CNN
        features = self.cnn(images)

        for name , param in self.cnn.named_parameters():
          if "fc.weight" in name or 'fc.bias' in name :
            param.requires_grad = True
          else :
            param.requires_grad = False

        # Project features to the specified embedding size
        features = self.projection(features)
        
        return features
    
    def get_feature_size(self):
        """Returns the raw feature size of the CNN backbone"""
        return self.feature_size