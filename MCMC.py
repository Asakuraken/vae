import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class SimpleEncoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20, hidden_dim=400):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        z = self.fc3(h)
        return z

class Decoder(nn.Module):
    """
    Simple decoder network that maps latent codes z to images x.
    Defines the likelihood p(x|z).
    """

    def __init__(self, latent_dim=20, hidden_dim=400):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 784)  # 28x28 = 784 for MNIST

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        x_logits = self.fc3(h)
        return x_logits


class MCMCBayesianModel:
    """
    Bayesian generative model with MCMC inference for posterior p(z|x).
    """

    def __init__(self, latent_dim=20, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.latent_dim = latent_dim
        self.device = device
        self.decoder = Decoder(latent_dim).to(device)
        self.encoder = SimpleEncoder(input_dim=784, latent_dim=self.latent_dim).to(self.device)
        # Prior: Standard Gaussian p(z) = N(0, I)
        self.prior_mean = torch.zeros(latent_dim).to(device)
        self.prior_std = torch.ones(latent_dim).to(device)

    def log_prior(self, z):
        """Log probability of prior p(z) = N(0, I)"""
        return -0.5 * torch.sum(z ** 2, dim=-1) - 0.5 * self.latent_dim * np.log(2 * np.pi)

    def log_likelihood(self, x, z):
        """
        Log likelihood p(x|z) using Bernoulli distribution.
        x: observed image [batch_size, 784]
        z: latent code [batch_size, latent_dim]
        """
        x_logits = self.decoder(z)
        # Bernoulli log likelihood
        log_lik = -F.binary_cross_entropy_with_logits(x_logits, x, reduction='none').sum(dim=-1)
        return log_lik

    def log_posterior_unnormalized(self, x, z):
        """Unnormalized log posterior: log p(z|x) ∝ log p(x|z) + log p(z)"""
        return self.log_likelihood(x, z) + self.log_prior(z)

    def metropolis_hastings_step(self, x, z_current, step_size=0.1):
        """
        Single Metropolis-Hastings step for sampling from p(z|x).
        """
        batch_size = z_current.shape[0]

        # Propose new z
        z_proposal = z_current + step_size * torch.randn_like(z_current)

        # Compute acceptance ratio
        log_p_current = self.log_posterior_unnormalized(x, z_current)
        log_p_proposal = self.log_posterior_unnormalized(x, z_proposal)
        log_alpha = log_p_proposal - log_p_current

        # Accept or reject
        u = torch.rand(batch_size).to(self.device)
        accept = (torch.log(u) < log_alpha).float().unsqueeze(1)

        z_new = accept * z_proposal + (1 - accept) * z_current

        return z_new, accept.mean().item()

    def sample_posterior(self, x, num_samples=1000, burn_in=500, step_size=0.1):
        """
        Sample from posterior p(z|x) using Metropolis-Hastings MCMC.
        Returns samples after burn-in period.
        """
        batch_size = x.shape[0]

        # Initialize z from prior
        z_current = torch.randn(batch_size, self.latent_dim).to(self.device)

        samples = []
        acceptance_rates = []

        # MCMC sampling
        for i in range(num_samples + burn_in):
            z_current, accept_rate = self.metropolis_hastings_step(x, z_current, step_size)

            if i >= burn_in:
                samples.append(z_current.clone())

            if i % 100 == 0:
                acceptance_rates.append(accept_rate)

        # Stack samples: [num_samples, batch_size, latent_dim]
        samples = torch.stack(samples)

        return samples, np.mean(acceptance_rates)

    def reconstruct(self, x, num_mcmc_samples=1000, burn_in=500):
        """
        Reconstruct images by sampling z from posterior and decoding.
        Returns mean reconstruction and samples.
        """
        with torch.no_grad():
            # Sample from posterior
            z_samples, _ = self.sample_posterior(x, num_mcmc_samples, burn_in)

            # Decode all samples
            reconstructions = []
            for z in z_samples:
                x_logits = self.decoder(z)
                x_probs = torch.sigmoid(x_logits)
                reconstructions.append(x_probs)

            reconstructions = torch.stack(reconstructions)

            # Mean reconstruction
            mean_reconstruction = reconstructions.mean(dim=0)

            # Sample reconstructions (for diversity visualization)
            sample_reconstructions = reconstructions[:5]  # First 5 samples

        return mean_reconstruction, sample_reconstructions

    def generate(self, num_samples=16):
        """Generate new images by sampling z from prior and decoding."""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            x_logits = self.decoder(z)
            x_probs = torch.sigmoid(x_logits)
        return x_probs

    def train_decoder(self, train_loader, epochs=50, lr=1e-3):
        """
        Pre-train decoder using samples from prior.
        This helps initialize the decoder for better MCMC convergence.
        """
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)

        print("Pre-training decoder...")
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.view(data.size(0), -1).to(self.device)

                # Sample z from prior
                z = torch.randn(data.size(0), self.latent_dim).to(self.device)

                # Decode and compute loss
                x_logits = self.decoder(z)

                # For pre-training, we'll use reconstruction on random pairs
                # This is a simple initialization strategy
                # target_idx = torch.randperm(data.size(0))
                # target = data[target_idx]
                target = data

                loss = F.binary_cross_entropy_with_logits(x_logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")


    def train_autoencoder(self, train_loader, epochs=50, lr=1e-3):
        """
        Jointly train encoder and decoder as a vanilla autoencoder
        """
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr)

        print("Training autoencoder...")
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.view(data.size(0), -1).to(self.device)

                # Encode to latent z
                z = self.encoder(data)

                # Decode to x'
                x_logits = self.decoder(z)

                # Compute reconstruction loss
                loss = F.binary_cross_entropy_with_logits(x_logits, data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")


def visualize_results(model, test_loader, device):
    """Visualize reconstructions and generated samples."""
    model.decoder.eval()

    # Get a batch of test images
    data, _ = next(iter(test_loader))
    data = data[:8].view(8, -1).to(device)

    # Reconstruct
    print("Running MCMC for reconstruction (this may take a while)...")
    start_time = time.time()
    mean_recon, sample_recons = model.reconstruct(data, num_mcmc_samples=500, burn_in=200)
    mcmc_time = time.time() - start_time
    print(f"MCMC reconstruction took {mcmc_time:.2f} seconds")

    # Generate new samples
    generated = model.generate(16)

    # Plotting
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))

    # Row 1: Original images
    for i in range(8):
        axes[0, i].imshow(data[i].cpu().view(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)

    # Row 2: Mean reconstructions
    for i in range(8):
        axes[1, i].imshow(mean_recon[i].cpu().view(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('MCMC Mean', fontsize=10)

    # Row 3-4: Generated samples
    for i in range(16):
        row = 2 + i // 8
        col = i % 8
        axes[row, col].imshow(generated[i].cpu().view(28, 28), cmap='gray')
        axes[row, col].axis('off')
        if i == 0:
            axes[row, col].set_title('Generated', fontsize=10)

    plt.tight_layout()
    plt.savefig('mcmc_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def analyze_mcmc_convergence(model, x, num_samples=2000, burn_in=0):
    """Analyze MCMC convergence by tracking log posterior over iterations."""
    batch_size = 1  # Analyze single example
    x = x[:batch_size]

    z_current = torch.randn(batch_size, model.latent_dim).to(model.device)
    log_posteriors = []
    z_trajectory = []

    for i in range(num_samples):
        z_current, _ = model.metropolis_hastings_step(x, z_current, step_size=0.1)
        log_post = model.log_posterior_unnormalized(x, z_current).item()
        log_posteriors.append(log_post)

        if i % 10 == 0:  # Save every 10th sample
            z_trajectory.append(z_current[0, :2].cpu().numpy())  # First 2 dims

    # Plot convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Log posterior trace
    ax1.plot(log_posteriors)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Log Posterior')
    ax1.set_title('MCMC Convergence')
    ax1.legend()

    # 2D trajectory (first 2 dimensions)
    z_trajectory = np.array(z_trajectory)
    ax2.plot(z_trajectory[:, 0], z_trajectory[:, 1], 'b-', alpha=0.5)
    ax2.scatter(z_trajectory[0, 0], z_trajectory[0, 1], c='g', s=100, label='Start')
    ax2.scatter(z_trajectory[-1, 0], z_trajectory[-1, 1], c='r', s=100, label='End')
    ax2.set_xlabel('z_1')
    ax2.set_ylabel('z_2')
    ax2.set_title('MCMC Trajectory (First 2 Dimensions)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('mcmc_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()

def compare_latent_dimensions_mcmc(train_loader, test_loader, device, dimensions=[2, 5, 10, 20, 50]):
    """Compare MCMC model reconstruction quality with different latent dimensions."""
    results = {}

    for dim in dimensions:
        print(f"\nEvaluating MCMC with latent dimension {dim}...")

        # Initialize model
        model = MCMCBayesianModel(latent_dim=dim, device=device)

        # Train autoencoder
        model.train_autoencoder(train_loader, epochs=30)

        # Get test batch
        data, _ = next(iter(test_loader))
        data = data[:64].view(64, -1).to(device)

        # Reconstruct using MCMC
        mean_recon, _ = model.reconstruct(data, num_mcmc_samples=500, burn_in=200)

        # Compute MSE
        mse = F.mse_loss(mean_recon, data).item()

        results[dim] = mse
        print(f"Latent dim {dim}: MSE = {mse:.4f}")

    # Plot results
    dims = list(results.keys())
    mses = [results[d] for d in dims]

    plt.figure(figsize=(8, 5))
    plt.plot(dims, mses, 'bo-')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Reconstruction MSE')
    plt.title('MCMC: Reconstruction Quality vs Latent Dimension')
    plt.grid(True)
    plt.savefig('mcmc_dimension_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results

def main():
    # Hyperparameters
    batch_size = 128
    latent_dim = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = MCMCBayesianModel(latent_dim=latent_dim, device=device)

    # Pre-train decoder (optional but helps convergence)
    # model.train_decoder(train_loader, epochs=50)
    model.train_autoencoder(train_loader, epochs=50)

    # Visualize results
    visualize_results(model, test_loader, device)

    # Analyze convergence
    test_data, _ = next(iter(test_loader))
    test_data = test_data.view(test_data.size(0), -1).to(device)
    analyze_mcmc_convergence(model, test_data)

    # Compare reconstruction quality
    print("\nComparing reconstruction quality...")
    data, _ = next(iter(test_loader))
    data = data[:16].view(16, -1).to(device)

    mean_recon, _ = model.reconstruct(data, num_mcmc_samples=1000, burn_in=500)

    # Compute reconstruction error
    mse = F.mse_loss(mean_recon, data).item()
    bce = F.binary_cross_entropy(mean_recon, data).item()

    print(f"Reconstruction MSE: {mse:.4f}")
    print(f"Reconstruction BCE: {bce:.4f}")

    # # Compare MCMC reconstruction quality with different latent dimensions
    # print("\nComparing MCMC reconstruction quality with different latent dimensions...")
    # compare_latent_dimensions_mcmc(train_loader, test_loader, device, dimensions=[20])

if __name__ == "__main__":
    main()