# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 18:08:35 2021

@author: Andreas
"""

import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
import torch
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


#Task 3 has included batch normalization and drop out
class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers

        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
    
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
           
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 4*4*128
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss

        self.classifier = nn.Sequential(
            nn.Linear(128*8*8, 64),
            nn.Dropout(p = 0.15),
            nn.LeakyReLU(),
            nn.Linear(64, self.num_classes),
            nn.Dropout(p = 0.15)
            
        )

        """
        img X

        self.feature_extractor(X) -> y
        
        """
        # The output of feature_extractor will be eonvolutional layersThere are 50,000 training images and 10, 000 test images.functions

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        
        batch_size = x.shape[0]
        features = self.feature_extractor(x)
        
        features = features.view((batch_size, -1))
        # print(features.shape)

        out = self.classifier(features)
        
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


    def save_model(self):
        def is_best_model():
            """
                Returns True if current model has the lowest validation loss
            """
            validation_losses = list(self.VALIDATION_LOSS.values())
            return validation_losses[-1] == min(validation_losses)

        state_dict = self.model.state_dict()
        filepath = self.checkpoint_dir.joinpath(f"{self.global_step}.ckpt")

        utils.save_checkpoint(state_dict, filepath, is_best_model())

    def load_best_model(self):
        state_dict = utils.load_best_checkpoint(self.checkpoint_dir)
        if state_dict is None:
            print(
                f"Could not load best checkpoint. Did not find under: {self.checkpoint_dir}")
            return
        self.model.load_state_dict(state_dict)



def create_plots(trainer2: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()
    plt.savefig(plot_path.joinpath(f"{name}_plot.eps"))


if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    print(torch.cuda.is_available())
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 8
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    create_plots(trainer, "task3_1_2_10_epochs")