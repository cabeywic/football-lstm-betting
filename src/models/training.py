import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def create_dataloader(X, y, batch_size=64):
    """
    Convert data to PyTorch tensors and create a DataLoader.
    
    Parameters:
    - X: Input data (numpy array).
    - y: Labels (numpy array).
    - batch_size: Batch size for the DataLoader.
    
    Returns:
    - DataLoader object.
    """
    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size)
    
    return loader

def prepare_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64):
    """
    Prepare DataLoader objects for training, validation, and test data.
    
    Parameters:
    - X_train, y_train: Training data and labels.
    - X_val, y_val: Validation data and labels.
    - X_test, y_test: Test data and labels.
    - batch_size: Batch size for the DataLoader.
    
    Returns:
    - train_loader, val_loader, test_loader: DataLoader objects for training, validation, and testing.
    """
    train_loader = create_dataloader(X_train, y_train, batch_size)
    val_loader = create_dataloader(X_val, y_val, batch_size)
    test_loader = create_dataloader(X_test, y_test, batch_size)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    # Initialize the tqdm progress bar
    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    # Lists to store average losses for each epoch
    train_losses = []
    val_losses = []
    
    for epoch in pbar:
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        for i, (sequences, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate average loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update tqdm progress bar with the average loss
        pbar.set_postfix({'Epoch Train Loss': f'{avg_train_loss:.4f}', 'Epoch Val Loss': f'{avg_val_loss:.4f}'})

    return train_losses, val_losses
    
def plot_loss_curves(train_losses, val_losses, title="Training and Validation Loss Curves"):
    """
    Plot training and validation loss curves.
    
    Parameters:
    - train_losses: List of training losses.
    - val_losses: List of validation losses.
    - title: Title for the plot (default is "Training and Validation Loss Curves").
    
    Returns:
    - A matplotlib plot displaying the loss curves.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()