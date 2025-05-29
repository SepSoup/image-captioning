#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Full caption model integrating encoder and decoder.
This module combines the CNN encoder and RNN decoder into a complete image captioning model.
"""

import torch
import torch.nn as nn
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN

class CaptionModel(nn.Module):
    """
    Complete image captioning model with CNN encoder and RNN decoder.
    """
    
    def __init__(self, 
             embed_size=256, 
             hidden_size=512, 
             vocab_size=10000, 
             num_layers=1,
             encoder_model='inception_v3',
             decoder_type='lstm',
             dropout=0.5,
             train_encoder=False):
        """
        Initialize the caption model.
        
        Args:
            embed_size (int): Dimensionality of the embedding space
            hidden_size (int): Dimensionality of the RNN hidden state
            vocab_size (int): Size of the vocabulary
            num_layers (int): Number of layers in the RNN
            encoder_model (str): Name of the CNN backbone for the encoder
            decoder_type (str): Type of RNN cell ('lstm' or 'gru')
            dropout (float): Dropout probability
            train_encoder (bool): Whether to fine-tune the encoder
        """
        #------------------------------------------------------------------------------------------------
        super(CaptionModel, self).__init__()
        
        self.encoderCNN = EncoderCNN()
        self.decoderRNN = DecoderRNN(embed_size,hidden_size,vocab_size, num_layers)

        # TO DO: Initialize the encoder and decoder components
        # 1. Create an EncoderCNN instance with the specified parameters ✅
        # 2. Create a DecoderRNN instance with the specified parameters  ✅
        #------------------------------------------------------------------------------------------------
        
    def forward(self, images, captions, hidden=None):
        """
        Forward pass for training with teacher forcing.
        
        Args:
            images (torch.Tensor): Input images [batch_size, 3, height, width]
            captions (torch.Tensor): Ground truth captions [batch_size, seq_length]
            hidden (tuple or torch.Tensor, optional): Initial hidden state for the RNN
            
        Returns:
            torch.Tensor: Output scores for each word in the vocabulary
                        Shape: [batch_size, seq_length, vocab_size]
            tuple or torch.Tensor: Final hidden state of the RNN
        """
        #------------------------------------------------------------------------------------------------
        features = self.encoderCNN(images)  # Extract image features [batch_size, embed_size]
        outputs, hidden = self.decoderRNN(features, captions, hidden)  # Pass features + captions to decoder
        return outputs, hidden
        # TO DO: Implement the forward pass of the full model
        # 1. Extract features from images using the encoder                                       ✅
        # 2. Use the decoder to generate captions based on the features and ground truth captions ✅
        # 3. Return the outputs and final hidden state                                            ✅
        #------------------------------------------------------------------------------------------------
        
        return outputs, hidden
    
    def generate_caption(self, image, max_length=20, start_token=1, end_token=2, beam_size=1):
        """
        Generate a caption for a single image.
        """
        device = image.device
        #------------------------------------------------------------------------------------------------
        # TODO: Implement caption generation for inference
        # 1. Extract features from the image using the encoder (with torch.no_grad()) ✅
        # 2. Use the decoder to generate a caption based on the features              ✅
        # 3. Return the generated caption                                             ✅
    
        # 1.
        with torch.no_grad():
            self.encoderCNN.eval()
            features = self.encoderCNN(image)  # [1, embed_size]
    
        # 2.
        sampled_ids = self.decoderRNN.sample(
            features,
            max_length=max_length,
            start_token=start_token,
            end_token=end_token,
            beam_size=beam_size
        )
    
        # 3.
        return sampled_ids[0]
        #------------------------------------------------------------------------------------------------
