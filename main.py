from model import VAE
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch


def arg_parse():
    parser = argparse.ArgumentParser(description="VAE Training Script")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    return parser.parse_args()

def main():
    args = arg_parse()


    train_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCELoss(reduction='sum')


    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device).view(-1, 784)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = criterion(recon_batch, data) + 0.5 * torch.sum(mu**2 + torch.exp(logvar) - logvar - 1)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}')
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device).view(-1, 784)
                recon_batch, mu, logvar = model(data)
                loss = criterion(recon_batch, data) + 0.5 * torch.sum(mu**2 + torch.exp(logvar) - logvar - 1)
                test_loss += loss.item()
        print(f'Test Loss: {test_loss / len(test_loader.dataset)}')
if __name__ == "__main__":
    main()

