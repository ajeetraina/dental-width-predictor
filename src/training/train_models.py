#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training script for dental radiograph analysis models.

This script handles the training of segmentation and classification models
for tooth detection and classification in dental radiographs.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the model definitions
from models.model import create_segmentation_model, create_tooth_classifier, save_trained_models


def create_data_generators(data_dir, batch_size=4, 
                          segmentation_shape=(512, 512), 
                          classification_shape=(128, 128)):
    """Create data generators for training the models.
    
    Args:
        data_dir (str): Directory containing training data
        batch_size (int): Batch size for training
        segmentation_shape (tuple): Input shape for segmentation model
        classification_shape (tuple): Input shape for classification model
        
    Returns:
        tuple: Data generators for segmentation and classification models
    """
    # Implement data generators according to your dataset structure
    # This is a placeholder and should be adapted to your specific data format
    
    # For segmentation model
    def segmentation_generator():
        while True:
            # Load a batch of images and masks
            batch_images = []
            batch_masks = []
            
            # Process images and masks
            for i in range(batch_size):
                # Load and preprocess image
                # ...
                
                # Load and preprocess mask
                # ...
                
                batch_images.append(image)
                batch_masks.append(mask)
            
            yield np.array(batch_images), np.array(batch_masks)
    
    # For classification model
    def classification_generator():
        while True:
            # Load a batch of tooth regions and labels
            batch_regions = []
            batch_labels = []
            
            # Process tooth regions and labels
            for i in range(batch_size):
                # Load and preprocess tooth region
                # ...
                
                # Load and preprocess label
                # ...
                
                batch_regions.append(region)
                batch_labels.append(label)
            
            yield np.array(batch_regions), np.array(batch_labels)
    
    return segmentation_generator(), classification_generator()


def train_segmentation_model(data_dir, model_dir, epochs=50, batch_size=4):
    """Train the tooth segmentation model.
    
    Args:
        data_dir (str): Directory containing training data
        model_dir (str): Directory to save the trained model
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tf.keras.Model: Trained segmentation model
    """
    # Create the model
    model = create_segmentation_model(input_shape=(512, 512, 1), num_classes=3)
    
    # Create data generators
    train_gen, val_gen = create_data_generators(data_dir, batch_size)
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        f"{model_dir}/segmentation_checkpoint.h5",
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        patience=10,
        monitor='val_loss',
        mode='min',
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        factor=0.2,
        patience=5,
        monitor='val_loss',
        mode='min',
        min_lr=1e-6
    )
    
    # Train the model
    history = model.fit(
        train_gen,
        steps_per_epoch=100,  # Adjust based on your dataset size
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=20,  # Adjust based on your dataset size
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.savefig(f"{model_dir}/segmentation_training_history.png")
    
    return model


def train_classification_model(data_dir, model_dir, epochs=50, batch_size=16):
    """Train the tooth classification model.
    
    Args:
        data_dir (str): Directory containing training data
        model_dir (str): Directory to save the trained model
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tf.keras.Model: Trained classification model
    """
    # Create the model
    model = create_tooth_classifier(input_shape=(128, 128, 1), num_classes=3)
    
    # Create data generators
    train_gen, val_gen = create_data_generators(data_dir, batch_size, 
                                              classification_shape=(128, 128))
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        f"{model_dir}/classification_checkpoint.h5",
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        patience=10,
        monitor='val_loss',
        mode='min',
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        factor=0.2,
        patience=5,
        monitor='val_loss',
        mode='min',
        min_lr=1e-6
    )
    
    # Train the model
    history = model.fit(
        train_gen,
        steps_per_epoch=100,  # Adjust based on your dataset size
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=20,  # Adjust based on your dataset size
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.savefig(f"{model_dir}/classification_training_history.png")
    
    return model


def main():
    """Main function for model training."""
    parser = argparse.ArgumentParser(description='Train dental radiograph analysis models')
    parser.add_argument('--data', type=str, required=True,
                        help='Directory containing training data')
    parser.add_argument('--models', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--segmentation_only', action='store_true',
                        help='Train only the segmentation model')
    parser.add_argument('--classification_only', action='store_true',
                        help='Train only the classification model')
    
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(args.models, exist_ok=True)
    
    # Train the models
    segmentation_model = None
    classification_model = None
    
    if not args.classification_only:
        print("\nTraining segmentation model...")
        segmentation_model = train_segmentation_model(
            args.data, args.models, args.epochs, args.batch_size
        )
    
    if not args.segmentation_only:
        print("\nTraining classification model...")
        classification_model = train_classification_model(
            args.data, args.models, args.epochs, args.batch_size
        )
    
    # Save the models
    if segmentation_model and classification_model:
        save_trained_models(segmentation_model, classification_model, args.models)
        print(f"\nModels saved to {args.models}")
    elif segmentation_model:
        segmentation_model.save(f"{args.models}/tooth_segmentation_model")
        print(f"\nSegmentation model saved to {args.models}/tooth_segmentation_model")
    elif classification_model:
        classification_model.save(f"{args.models}/tooth_classification_model")
        print(f"\nClassification model saved to {args.models}/tooth_classification_model")


if __name__ == "__main__":
    main()