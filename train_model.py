from tqdm import tqdm
import torch

def train_model(topo_model, optimizer, train_loader, val_loader, test_loader, device='cuda', n_epochs=100):
    
    train_losses = []
    val_losses = []
    progress = tqdm(range(n_epochs))
    
    for epoch in progress:
        # Training phase
        topo_model.train()
        train_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            
            optimizer.zero_grad()
            loss = topo_model(x)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
    
        # Calculate average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        topo_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                loss = topo_model(x)
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update progress bar
        progress.set_postfix({
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        })
        
        # Print every 5 epochs
        if epoch % 5 == 0:
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Final evaluation on test set
    topo_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            loss = topo_model(x)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f'\nFinal Test Loss: {avg_test_loss:.4f}')

    loss_history = {
        'training': train_losses,
        'validation': val_losses
    }
    
    return topo_model, loss_history
    