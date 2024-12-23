import torch
import numpy as np


class EarlyStopping:
    '''
    validation lossがpatience回以上更新されなければself.early_stopをTrueに
    '''
    def __init__(self, patience, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.early_stop = False
        self.min_value = np.Inf
    def __call__(self, value):
        if value >= self.min_value-self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            print(f'Validation loss decreased ({self.min_value} --> {value})')
            self.min_value = value


def train(dataloader, model, criterion, optimizer, device):
    '''Train the model for one epoch and calculate average loss.'''
    model.train()
    epoch_loss = 0.0
    num_samples = 0

    for data in dataloader:
        # Forward pass and loss computation
        input_imgs = data.to(device)
        output_imgs = model(input_imgs)
        loss = criterion(output_imgs, input_imgs)

        # Backward pass and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss and count samples
        batch_size_actual = data.size(0)
        epoch_loss += loss.item() * batch_size_actual
        num_samples += batch_size_actual

    # Return average loss for the epoch
    return epoch_loss / num_samples


def val(dataloader, model, criterion, device):
    '''Validate the model and calculate average loss.'''
    model.eval()
    epoch_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for data in dataloader:
            # Forward pass and loss computation
            input_imgs = data.to(device)
            output_imgs = model(input_imgs)
            loss = criterion(output_imgs, input_imgs)

            # Accumulate loss and count samples
            batch_size = input_imgs.size(0)
            epoch_loss += loss.item() * batch_size
            num_samples += batch_size

    # Return average loss for the validation dataset
    return epoch_loss / num_samples
