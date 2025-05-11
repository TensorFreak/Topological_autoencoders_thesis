import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd


class CSVDataset(Dataset):
    
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.coordinates = torch.tensor(self.data[['x', 'y']].values, dtype=torch.float32)
        self.data.drop(['time_s', 'x', 'y'], axis=1, inplace=True)
        self.data = torch.tensor(self.data.values, dtype=torch.float32)
        self.dimension = self.data.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.coordinates[idx]
        

def loaders(data_path='neural_data/22ht1.csv', train_size=0.7, val_size=0.2, batch_size=128, data_seed=42):
    
    mice_data = CSVDataset(data_path)
    # Define split sizes
    train_size = int(train_size * len(mice_data))  
    val_size = int(val_size * len(mice_data)) 
    test_size = len(mice_data) - train_size - val_size 
    
    train_data, val_data, test_data = random_split(
        mice_data, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(data_seed)  # for reproducibility
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
    
    print(f'Total data samples: {len(mice_data)}')
    print(f'Train samples: {len(train_data)} ({len(train_data)/len(mice_data):.1%})')
    print(f'Validation samples: {len(val_data)} ({len(val_data)/len(mice_data):.1%})')
    print(f'Test samples: {len(test_data)} ({len(test_data)/len(mice_data):.1%})')

    return train_loader, val_loader, test_loader