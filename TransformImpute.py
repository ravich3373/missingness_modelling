import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

class SimpleTransformer(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(SimpleTransformer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
        )
        self.transformer = nn.Transformer(hidden_size, nhead=4, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout=0.1)
        self.out_layers = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),  # Add dropout layer
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.transformer(x.unsqueeze(1), x.unsqueeze(1)).squeeze(1)
        x = self.out_layers(x)
        return x

def preprocess_data(data):
    # Drop columns with more than 75% missing values
    data = data.loc[:, data.isnull().mean() < 0.75].copy()

    # Separate the numeric and non-numeric columns
    numeric_columns = data.select_dtypes(include=np.number).columns
    non_numeric_columns = data.select_dtypes(exclude=np.number).columns

    # Normalize the numeric columns
    
    scaler = StandardScaler().fit(data[numeric_columns])
    data[numeric_columns] = scaler.transform(data[numeric_columns])
    '''
    eps = 1e-8  # Small constant to avoid division by zero or too small numbers
    for column in numeric_columns:
        col_min = data[column].min(skipna=True)
        col_max = data[column].max(skipna=True)
        data[column] = (data[column] - col_min) / (col_max - col_min + eps)
    '''

    # Encode the non-numeric columns
    #for i, column in enumerate(non_numeric_columns):
       #data[data.columns[i]] = pd.Categorical(data[column]).codes.astype('float64')
    # This didn't work so for now, we just drop non-numeric columns
    data = data.drop(non_numeric_columns, axis=1)

    # Convert all remaining columns to a numeric type
    data = data.apply(pd.to_numeric)

    return data

def mask_random_values(tensor, mask_fraction):
    mask = torch.rand(tensor.shape) < mask_fraction
    masked_tensor = tensor.clone().detach()
    #masked_tensor[mask] = float('nan')
    masked_tensor[mask] = 0
    return masked_tensor, mask

def masked_mse_loss(output, target, mask):
    mse = (output - target)**2
    masked_mse = mse * mask
    return masked_mse.sum() / mask.sum()

def main():
    # Load the data
    data = pd.read_csv('ADNIMERGE.csv', low_memory=False)
    data = preprocess_data(data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split the data into train, validation, and test sets
    train_data, val_data, test_data = np.split(data.sample(frac=1, random_state=42),
                                               [int(.6*len(data)), int(.8*len(data))])

    train_data, val_data, test_data = [torch.tensor(d.values, dtype=torch.float32).to(device) for d in [train_data, val_data, test_data]]

    # Set hyperparameters and other configurations
    num_columns = data.shape[1]
    hidden_size = 64
    num_layers = 3
    learning_rate = 0.001
    num_epochs = 5

    # Create DataLoader for training, validation, and test sets
    train_loader = DataLoader(TensorDataset(train_data), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=32, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = SimpleTransformer(num_columns, hidden_size=hidden_size, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training and validation loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader):
            x = batch[0].to(device)
            x = torch.nan_to_num(x)
            # Mask random values in the input data
            masked_x, mask = mask_random_values(x, mask_fraction=0.1)

            optimizer.zero_grad()
            output = model(masked_x)

            # Gradually replace masked values with model predictions
            for _ in range(5):
                output = model(masked_x)
                updated_masked_x = masked_x.clone()
                updated_masked_x[mask] = output[mask]
                masked_x = updated_masked_x
            
            #print("Input data:", x)
            #print("Masked data:", masked_x)
            #print("Model output:", output)

            # Calculate the loss only for masked values
            train_loss = masked_mse_loss(output, x, mask)
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Training MSE: {total_train_loss / len(train_loader)}")

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                mask = x.isnan()
                x = torch.nan_to_num(x)
                output = model(x)
                val_loss = masked_mse_loss(output, x, mask)
                total_val_loss += val_loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Validation MSE: {total_val_loss / len(val_loader)}")

    # Test the model
    model.eval()
    total_test_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            mask = x.isnan()
            x = torch.nan_to_num(x)
            output = model(x)
            test_loss = masked_mse_loss(output, x, mask)
            total_test_loss += test_loss.item()

    print(f"Test MSE on missing values: {total_test_loss / len(test_loader)}")

    # Save the model's state
    model_save_path = "model_state.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model's state saved to {model_save_path}")

main()
