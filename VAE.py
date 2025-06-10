import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class Encoder(nn.Module):
    """
    Encoder network Q(z|x) that maps images to latent distribution parameters.
    """

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder network P(x|z) that maps latent codes to images.
    """

    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        x_logits = self.fc3(h)
        return x_logits


class VAE(nn.Module):
    """
    Variational Autoencoder combining encoder and decoder.
    """

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        where epsilon ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        mu, logvar = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x_logits = self.decoder(z)

        return x_logits, mu, logvar, z

    def decode(self, z):
        """Decode latent codes to images."""
        return self.decoder(z)

    def encode(self, x):
        """Encode images to latent distribution parameters."""
        return self.encoder(x)


def vae_loss(x_logits, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence

    Args:
        x_logits: Decoder output logits
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL term (beta-VAE)
    """
    # Reconstruction loss (Bernoulli)
    recon_loss = F.binary_cross_entropy_with_logits(
        x_logits, x, reduction='sum'
    ) / x.size(0)

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # Total loss (ELBO)
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def train_vae(model, train_loader, test_loader, epochs=100, lr=1e-3, device='cuda'):
    """Train the VAE model."""
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0

        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')):
            data = data.view(data.size(0), -1).to(device)

            optimizer.zero_grad()

            # Forward pass
            x_logits, mu, logvar, _ = model(data)

            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(x_logits, data, mu, logvar)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()

        # Evaluation
        model.eval()
        test_loss = 0
        test_recon_loss = 0
        test_kl_loss = 0

        with torch.no_grad():
            for data, _ in test_loader:
                data = data.view(data.size(0), -1).to(device)
                x_logits, mu, logvar, _ = model(data)
                loss, recon_loss, kl_loss = vae_loss(x_logits, data, mu, logvar)

                test_loss += loss.item()
                test_recon_loss += recon_loss.item()
                test_kl_loss += kl_loss.item()

        # Average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        if (epoch + 1) % 10 == 0:
            print(f'\nEpoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
            print(
                f'Train Recon: {train_recon_loss / len(train_loader):.4f}, Train KL: {train_kl_loss / len(train_loader):.4f}')

    return train_losses, test_losses


def visualize_reconstructions(model, test_loader, device, num_samples=10):
    """Visualize original images and their reconstructions."""
    model.eval()

    # Get a batch of test images
    data, labels = next(iter(test_loader))
    data = data[:num_samples].to(device)

    with torch.no_grad():
        # Reconstruct
        x_flat = data.view(data.size(0), -1)
        x_logits, _, _, z = model(x_flat)
        x_recon = torch.sigmoid(x_logits).view(-1, 28, 28)

    # Plot
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 4))

    for i in range(num_samples):
        # Original
        axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')

        # Reconstruction
        axes[1, i].imshow(x_recon[i].cpu(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstruction')

    plt.tight_layout()
    plt.savefig('vae_reconstructions.png', dpi=150, bbox_inches='tight')
    plt.show()


def generate_samples(model, num_samples=100, device='cuda'):
    """Generate new samples from the learned distribution."""
    model.eval()

    with torch.no_grad():
        # Sample from prior N(0, I)
        z = torch.randn(num_samples, model.latent_dim).to(device)

        # Decode
        x_logits = model.decode(z)
        x_samples = torch.sigmoid(x_logits).view(-1, 28, 28)

    # Visualize grid of samples
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    axes = axes.ravel()

    for i in range(100):
        axes[i].imshow(x_samples[i].cpu(), cmap='gray')
        axes[i].axis('off')

    plt.suptitle('Generated Samples from VAE', fontsize=16)
    plt.tight_layout()
    plt.savefig('vae_generated_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

    return x_samples


def analyze_latent_space(model, test_loader, device):
    """Analyze the learned latent space."""
    model.eval()

    latent_codes = []
    labels_list = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.view(data.size(0), -1).to(device)
            mu, _ = model.encode(data)
            latent_codes.append(mu.cpu())
            labels_list.append(labels)

    latent_codes = torch.cat(latent_codes, dim=0).numpy()
    labels_list = torch.cat(labels_list, dim=0).numpy()

    # Visualize 2D latent space (if latent_dim >= 2)
    if model.latent_dim >= 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_codes[:, 0], latent_codes[:, 1],
                              c=labels_list, cmap='tab10', alpha=0.5, s=2)
        plt.colorbar(scatter)
        plt.xlabel('z_1')
        plt.ylabel('z_2')
        plt.title('Latent Space Visualization (First 2 Dimensions)')
        plt.savefig('vae_latent_space.png', dpi=150, bbox_inches='tight')
        plt.show()

    # Analyze latent dimensions usage
    latent_std = np.std(latent_codes, axis=0)
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(latent_std)), latent_std)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Standard Deviation')
    plt.title('Latent Dimension Usage (Higher std = more active)')
    plt.savefig('vae_latent_usage.png', dpi=150, bbox_inches='tight')
    plt.show()

    return latent_codes, labels_list


def interpolate_latent(model, test_loader, device, digit1=0, digit2=8, steps=10):
    """Interpolate between two digits in latent space."""
    model.eval()

    # Find examples of each digit
    data1, data2 = None, None

    for data, labels in test_loader:
        if data1 is None:
            idx1 = (labels == digit1).nonzero(as_tuple=True)[0]
            if len(idx1) > 0:
                data1 = data[idx1[0]].view(1, -1).to(device)

        if data2 is None:
            idx2 = (labels == digit2).nonzero(as_tuple=True)[0]
            if len(idx2) > 0:
                data2 = data[idx2[0]].view(1, -1).to(device)

        if data1 is not None and data2 is not None:
            break

    with torch.no_grad():
        # Encode
        mu1, _ = model.encode(data1)
        mu2, _ = model.encode(data2)

        # Interpolate
        interpolations = []
        for alpha in np.linspace(0, 1, steps):
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            x_logits = model.decode(z_interp)
            x_interp = torch.sigmoid(x_logits).view(28, 28)
            interpolations.append(x_interp.cpu())

    # Visualize
    fig, axes = plt.subplots(1, steps, figsize=(15, 2))
    for i, img in enumerate(interpolations):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

    plt.suptitle(f'Latent Space Interpolation: {digit1} → {digit2}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'vae_interpolation_{digit1}_to_{digit2}.png', dpi=150, bbox_inches='tight')
    plt.show()


def analyze_failure_modes(model, test_loader, device, num_failures=20):
    """Identify and visualize failure modes."""
    model.eval()

    # Collect reconstruction errors
    errors = []
    originals = []
    reconstructions = []

    with torch.no_grad():
        for data, _ in test_loader:
            data_flat = data.view(data.size(0), -1).to(device)
            x_logits, _, _, _ = model(data_flat)
            x_recon = torch.sigmoid(x_logits)

            # Compute per-sample reconstruction error
            mse = F.mse_loss(x_recon, data_flat, reduction='none').mean(dim=1)

            errors.extend(mse.cpu().numpy())
            originals.extend(data.cpu())
            reconstructions.extend(x_recon.view(-1, 28, 28).cpu())

    # Find worst reconstructions
    errors = np.array(errors)
    worst_idx = np.argsort(errors)[-num_failures:]

    # Visualize failures
    fig, axes = plt.subplots(2, num_failures // 2, figsize=(15, 6))
    axes = axes.ravel()

    for i, idx in enumerate(worst_idx[:num_failures // 2]):
        # Show original and reconstruction side by side
        ax_orig = axes[2 * i]
        ax_recon = axes[2 * i + 1]

        ax_orig.imshow(originals[idx].squeeze(), cmap='gray')
        ax_orig.set_title(f'Original\n(Error: {errors[idx]:.4f})')
        ax_orig.axis('off')

        ax_recon.imshow(reconstructions[idx], cmap='gray')
        ax_recon.set_title('Reconstruction')
        ax_recon.axis('off')

    plt.suptitle('VAE Failure Modes (Worst Reconstructions)', fontsize=16)
    plt.tight_layout()
    plt.savefig('vae_failure_modes.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Analyze ambiguous generations
    analyze_ambiguous_generations(model, device)


def analyze_ambiguous_generations(model, device, num_samples=50):
    """Analyze ambiguous or unrealistic generated samples."""
    model.eval()

    with torch.no_grad():
        # Generate many samples
        z = torch.randn(num_samples, model.latent_dim).to(device)
        x_logits = model.decode(z)
        x_samples = torch.sigmoid(x_logits).view(-1, 28, 28)

        # Compute "ambiguity" metric (entropy of pixel values)
        # High entropy = more uncertain/blurry
        entropy = -x_samples * torch.log(x_samples + 1e-8) - (1 - x_samples) * torch.log(1 - x_samples + 1e-8)
        entropy = entropy.mean(dim=(1, 2))

        # Find most ambiguous samples
        ambiguous_idx = torch.argsort(entropy, descending=True)[:10]

    # Visualize ambiguous samples
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.ravel()

    for i, idx in enumerate(ambiguous_idx):
        axes[i].imshow(x_samples[idx].cpu(), cmap='gray')
        axes[i].set_title(f'Entropy: {entropy[idx]:.3f}')
        axes[i].axis('off')

    plt.suptitle('Ambiguous Generated Samples (High Entropy)', fontsize=14)
    plt.tight_layout()
    plt.savefig('vae_ambiguous_samples.png', dpi=150, bbox_inches='tight')
    plt.show()


def compare_latent_dimensions(train_loader, test_loader, device, dimensions=[2, 5, 10, 20, 50]):
    """Compare VAE performance with different latent dimensions."""
    results = {}

    for dim in dimensions:
        print(f"\nTraining VAE with latent dimension {dim}...")

        # Create and train model
        model = VAE(latent_dim=dim).to(device)
        train_losses, test_losses = train_vae(model, train_loader, test_loader,
                                              epochs=50, device=device)

        # Evaluate reconstruction quality
        model.eval()
        total_mse = 0
        total_samples = 0

        with torch.no_grad():
            for data, _ in test_loader:
                data_flat = data.view(data.size(0), -1).to(device)
                x_logits, _, _, _ = model(data_flat)
                x_recon = torch.sigmoid(x_logits)

                mse = F.mse_loss(x_recon, data_flat, reduction='sum')
                total_mse += mse.item()
                total_samples += data.size(0)

        avg_mse = total_mse / total_samples

        results[dim] = {
            'model': model,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'mse': avg_mse,
            'final_test_loss': test_losses[-1]
        }

        print(f"Latent dim {dim}: MSE = {avg_mse:.4f}, Final test loss = {test_losses[-1]:.4f}")

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # MSE vs latent dimension
    dims = list(results.keys())
    mses = [results[d]['mse'] for d in dims]
    ax1.plot(dims, mses, 'bo-')
    ax1.set_xlabel('Latent Dimension')
    ax1.set_ylabel('Reconstruction MSE')
    ax1.set_title('Reconstruction Quality vs Latent Dimension')
    ax1.grid(True)

    # Test loss curves
    for dim in dimensions:
        ax2.plot(results[dim]['test_losses'], label=f'dim={dim}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Test Loss During Training')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('vae_dimension_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 128
    latent_dim = 20
    epochs = 100
    lr = 1e-3

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create and train model
    print("Training standard VAE...")
    model = VAE(latent_dim=latent_dim).to(device)
    train_losses, test_losses = train_vae(model, train_loader, test_loader, epochs=epochs, lr=lr, device=device)

    # Save model
    torch.save(model.state_dict(), 'vae_model.pth')

    # Visualizations and analysis
    print("\nGenerating visualizations...")

    # 1. Reconstruction quality
    visualize_reconstructions(model, test_loader, device)

    # 2. Generate new samples
    generated_samples = generate_samples(model, num_samples=100, device=device)

    # 3. Analyze latent space
    latent_codes, labels = analyze_latent_space(model, test_loader, device)

    # 4. Latent space interpolation
    interpolate_latent(model, test_loader, device, digit1=3, digit2=8)
    interpolate_latent(model, test_loader, device, digit1=4, digit2=9)

    # 5. Analyze failure modes
    analyze_failure_modes(model, test_loader, device)

    # 6. Compare different latent dimensions
    print("\nComparing different latent dimensions...")
    dimension_results = compare_latent_dimensions(train_loader, test_loader, device)

    # 7. Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('vae_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nVAE training and analysis complete!")
    print(f"Final test loss: {test_losses[-1]:.4f}")


if __name__ == "__main__":
    main()