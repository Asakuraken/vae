import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


class ModifiedMNISTDataset(Dataset):
    """
    Modified MNIST dataset where:
    - X: binarized central column of pixels (conditioning input)
    - Y: complete digit image (target output)
    """

    def __init__(self, mnist_dataset, threshold=0.5):
        self.mnist_dataset = mnist_dataset
        self.threshold = threshold

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx]
        image = image.squeeze()  # Remove channel dimension

        # Extract central column (column 14 for 28x28 images)
        central_column = image[:, 14]

        # Binarize the central column
        x_condition = (central_column > self.threshold).float()

        # Y is the complete image (flattened)
        y_target = image.view(-1)

        return x_condition, y_target, label


class CVAEEncoder(nn.Module):
    """
    Conditional encoder Q(z|Y, X) that takes both the target Y and condition X.
    """

    def __init__(self, y_dim=784, x_dim=28, hidden_dim=400, latent_dim=20):
        super(CVAEEncoder, self).__init__()

        # Concatenate Y and X as input
        input_dim = y_dim + x_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, y, x):
        # Concatenate y and x
        inputs = torch.cat([y, x], dim=1)

        h = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(h))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class CVAEDecoder(nn.Module):
    """
    Conditional decoder P(Y|z, X) that takes both latent z and condition X.
    """

    def __init__(self, latent_dim=20, x_dim=28, hidden_dim=400, y_dim=784):
        super(CVAEDecoder, self).__init__()

        # Concatenate z and X as input
        input_dim = latent_dim + x_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, y_dim)

    def forward(self, z, x):
        # Concatenate z and x
        inputs = torch.cat([z, x], dim=1)

        h = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(h))
        y_logits = self.fc3(h)

        return y_logits


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder.
    """

    def __init__(self, y_dim=784, x_dim=28, hidden_dim=400, latent_dim=20):
        super(CVAE, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = CVAEEncoder(y_dim, x_dim, hidden_dim, latent_dim)
        self.decoder = CVAEDecoder(latent_dim, x_dim, hidden_dim, y_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, y, x):
        # Encode
        mu, logvar = self.encoder(y, x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        y_logits = self.decoder(z, x)

        return y_logits, mu, logvar

    def sample(self, x, num_samples=1):
        """
        Sample multiple outputs for a given conditioning input x.
        """
        batch_size = x.size(0)
        samples = []

        with torch.no_grad():
            for _ in range(num_samples):
                # Sample z from prior N(0, I)
                z = torch.randn(batch_size, self.latent_dim).to(x.device)

                # Decode with conditioning
                y_logits = self.decoder(z, x)
                y_probs = torch.sigmoid(y_logits)
                samples.append(y_probs)

        return samples


class RegressionBaseline(nn.Module):
    """
    Simple regression model that directly maps X to Y.
    This will tend to average over all possible outputs.
    """

    def __init__(self, x_dim=28, hidden_dim=400, y_dim=784):
        super(RegressionBaseline, self).__init__()

        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, y_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        y_logits = self.fc4(h)
        return y_logits


def cvae_loss(y_logits, y, mu, logvar, beta=1.0):
    """CVAE loss function."""
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy_with_logits(
        y_logits, y, reduction='sum'
    ) / y.size(0)

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / y.size(0)

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def train_cvae(model, train_loader, test_loader, epochs=100, lr=1e-3, device='cuda'):
    """Train the CVAE model."""
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    for epoch in tqdm(range(epochs)):
        # Training
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0

        for x_cond, y_target, _ in train_loader:
            x_cond = x_cond.to(device)
            y_target = y_target.to(device)

            optimizer.zero_grad()

            # Forward pass
            y_logits, mu, logvar = model(y_target, x_cond)

            # Loss
            loss, recon_loss, kl_loss = cvae_loss(y_logits, y_target, mu, logvar)

            # Backward
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()

        # Evaluation
        model.eval()
        test_loss = 0

        with torch.no_grad():
            for x_cond, y_target, _ in test_loader:
                x_cond = x_cond.to(device)
                y_target = y_target.to(device)

                y_logits, mu, logvar = model(y_target, x_cond)
                loss, _, _ = cvae_loss(y_logits, y_target, mu, logvar)

                test_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        if (epoch + 1) % 10 == 0:
            print(f'\nEpoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
            print(f'Recon: {train_recon / len(train_loader):.4f}, KL: {train_kl / len(train_loader):.4f}')

    return train_losses, test_losses


def train_regression(model, train_loader, test_loader, epochs=100, lr=1e-3, device='cuda'):
    """Train the regression baseline."""
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0

        for x_cond, y_target, _ in train_loader:
            x_cond = x_cond.to(device)
            y_target = y_target.to(device)

            optimizer.zero_grad()

            # Forward pass
            y_logits = model(x_cond)

            # Loss
            loss = F.binary_cross_entropy_with_logits(y_logits, y_target)

            # Backward
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Evaluation
        model.eval()
        test_loss = 0

        with torch.no_grad():
            for x_cond, y_target, _ in test_loader:
                x_cond = x_cond.to(device)
                y_target = y_target.to(device)

                y_logits = model(x_cond)
                loss = F.binary_cross_entropy_with_logits(y_logits, y_target)

                test_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

    return train_losses, test_losses


def visualize_ambiguous_completions(cvae_model, regression_model, test_loader, device, num_examples=5):
    """
    Visualize how CVAE produces diverse completions while regression averages.
    """
    cvae_model.eval()
    regression_model.eval()

    # Get test examples
    x_conds = []
    y_targets = []
    labels = []

    for x_cond, y_target, label in test_loader:
        x_conds.append(x_cond)
        y_targets.append(y_target)
        labels.append(label)
        if len(x_conds) >= num_examples:
            break

    x_conds = torch.cat(x_conds)[:num_examples].to(device)
    y_targets = torch.cat(y_targets)[:num_examples].to(device)
    labels = torch.cat(labels)[:num_examples]

    # Generate samples
    num_cvae_samples = 5

    fig, axes = plt.subplots(num_examples, num_cvae_samples + 3, figsize=(15, num_examples * 2))

    with torch.no_grad():
        for i in range(num_examples):
            x_cond_i = x_conds[i:i + 1]
            y_target_i = y_targets[i:i + 1]

            # Column 0: Central column input (conditioning)
            central_col_vis = x_cond_i.cpu().numpy().reshape(-1, 1)
            central_col_vis = np.repeat(central_col_vis, 28, axis=1)  # Make it visible
            axes[i, 0].imshow(central_col_vis, cmap='gray')
            axes[i, 0].set_title('Input Column' if i == 0 else '')
            axes[i, 0].axis('off')

            # Column 1: Ground truth
            axes[i, 1].imshow(y_target_i.cpu().view(28, 28), cmap='gray')
            axes[i, 1].set_title('Ground Truth' if i == 0 else '')
            axes[i, 1].axis('off')

            # Column 2: Regression output
            reg_output = torch.sigmoid(regression_model(x_cond_i))
            axes[i, 2].imshow(reg_output.cpu().view(28, 28), cmap='gray')
            axes[i, 2].set_title('Regression' if i == 0 else '')
            axes[i, 2].axis('off')

            # Columns 3+: CVAE samples
            cvae_samples = cvae_model.sample(x_cond_i, num_samples=num_cvae_samples)
            for j, sample in enumerate(cvae_samples):
                axes[i, j + 3].imshow(sample.cpu().view(28, 28), cmap='gray')
                axes[i, j + 3].set_title(f'CVAE Sample {j + 1}' if i == 0 else '')
                axes[i, j + 3].axis('off')

    plt.suptitle('CVAE vs Regression: Handling Ambiguous Inputs', fontsize=16)
    plt.tight_layout()
    plt.savefig('cvae_vs_regression_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def analyze_diversity(cvae_model, test_loader, device, num_conditions=10, samples_per_condition=20):
    """
    Analyze the diversity of CVAE outputs for the same conditioning input.
    """
    cvae_model.eval()

    # Collect examples with different conditioning inputs
    x_conds = []

    for x_cond, _, _ in test_loader:
        x_conds.append(x_cond)
        if len(x_conds) >= num_conditions:
            break

    x_conds = torch.cat(x_conds)[:num_conditions].to(device)

    # For each conditioning input, generate multiple samples and compute diversity
    diversity_scores = []

    with torch.no_grad():
        for i in range(num_conditions):
            x_cond_i = x_conds[i:i + 1]

            # Generate multiple samples
            samples = cvae_model.sample(x_cond_i, num_samples=samples_per_condition)
            samples_tensor = torch.stack(samples).squeeze(1)  # [num_samples, 784]

            # Compute pairwise distances as diversity metric
            distances = []
            for j in range(samples_per_condition):
                for k in range(j + 1, samples_per_condition):
                    dist = F.mse_loss(samples_tensor[j], samples_tensor[k])
                    distances.append(dist.item())

            avg_diversity = np.mean(distances) if distances else 0
            diversity_scores.append(avg_diversity)

    # Visualize diversity analysis
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_conditions), diversity_scores)
    plt.xlabel('Conditioning Input Index')
    plt.ylabel('Average Pairwise MSE Between Samples')
    plt.title('CVAE Output Diversity for Different Conditioning Inputs')
    plt.grid(True, alpha=0.3)
    plt.savefig('cvae_diversity_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Average diversity score: {np.mean(diversity_scores):.4f}")

    return diversity_scores


def find_ambiguous_cases(cvae_model, test_loader, device, num_cases=5):
    """
    Find cases where the same central column corresponds to different digits.
    """
    cvae_model.eval()

    # Group data by central column pattern
    column_to_samples = defaultdict(list)

    for x_cond, y_target, label in test_loader:
        for i in range(len(x_cond)):
            # Convert to tuple for hashing
            column_key = tuple(x_cond[i].numpy().astype(int))
            column_to_samples[column_key].append({
                'y_target': y_target[i],
                'label': label[i].item(),
                'x_cond': x_cond[i]
            })

    # Find columns that map to multiple different digits
    ambiguous_cases = []
    for column_key, samples in column_to_samples.items():
        unique_labels = set(s['label'] for s in samples)
        if len(unique_labels) > 1:
            ambiguous_cases.append({
                'column': column_key,
                'samples': samples[:5],  # Keep first 5 examples
                'labels': unique_labels
            })

    # Sort by number of different labels
    ambiguous_cases.sort(key=lambda x: len(x['labels']), reverse=True)

    # Visualize most ambiguous cases
    fig, axes = plt.subplots(min(num_cases, len(ambiguous_cases)), 8,
                             figsize=(16, min(num_cases, len(ambiguous_cases)) * 2))

    if len(axes.shape) == 1:
        axes = axes.reshape(1, -1)

    for case_idx, case in enumerate(ambiguous_cases[:num_cases]):
        x_cond = case['samples'][0]['x_cond'].unsqueeze(0).to(device)

        # Show central column
        central_col_vis = x_cond.cpu().numpy().reshape(-1, 1)
        central_col_vis = np.repeat(central_col_vis, 28, axis=1)
        axes[case_idx, 0].imshow(central_col_vis, cmap='gray')
        axes[case_idx, 0].set_title(f"Labels: {sorted(case['labels'])}" if case_idx == 0 else '')
        axes[case_idx, 0].axis('off')

        # Show ground truth examples
        for i, sample in enumerate(case['samples'][:3]):
            axes[case_idx, i + 1].imshow(sample['y_target'].view(28, 28), cmap='gray')
            axes[case_idx, i + 1].set_title(f"GT: {sample['label']}" if case_idx == 0 else f"{sample['label']}")
            axes[case_idx, i + 1].axis('off')

        # Show CVAE samples
        with torch.no_grad():
            cvae_samples = cvae_model.sample(x_cond, num_samples=4)
            for i, sample in enumerate(cvae_samples):
                axes[case_idx, i + 4].imshow(sample.cpu().view(28, 28), cmap='gray')
                axes[case_idx, i + 4].set_title(f'CVAE {i + 1}' if case_idx == 0 else '')
                axes[case_idx, i + 4].axis('off')

    plt.suptitle('Ambiguous Cases: Same Central Column, Different Digits', fontsize=16)
    plt.tight_layout()
    plt.savefig('cvae_ambiguous_cases.png', dpi=150, bbox_inches='tight')
    plt.show()

    return ambiguous_cases


def compare_models_quantitatively(cvae_model, regression_model, test_loader, device):
    """
    Quantitative comparison between CVAE and regression.
    """
    cvae_model.eval()
    regression_model.eval()

    cvae_mse = 0
    regression_mse = 0
    cvae_samples_mse = []

    total_samples = 0

    with torch.no_grad():
        for x_cond, y_target, _ in test_loader:
            x_cond = x_cond.to(device)
            y_target = y_target.to(device)

            # Regression prediction
            reg_output = torch.sigmoid(regression_model(x_cond))
            reg_mse = F.mse_loss(reg_output, y_target, reduction='sum')
            regression_mse += reg_mse.item()

            # CVAE samples (average over multiple samples)
            cvae_samples = cvae_model.sample(x_cond, num_samples=10)

            # Best sample MSE (oracle - what if we could pick the best)
            best_mses = []
            for i in range(len(x_cond)):
                sample_mses = [F.mse_loss(s[i], y_target[i]).item() for s in cvae_samples]
                best_mses.append(min(sample_mses))

            cvae_samples_mse.extend(best_mses)

            # Average CVAE reconstruction
            avg_cvae = torch.stack(cvae_samples).mean(dim=0)
            avg_mse = F.mse_loss(avg_cvae, y_target, reduction='sum')
            cvae_mse += avg_mse.item()

            total_samples += len(x_cond)

    # Compute averages
    avg_regression_mse = regression_mse / total_samples
    avg_cvae_mse = cvae_mse / total_samples
    avg_best_cvae_mse = np.mean(cvae_samples_mse)

    print("\n=== Quantitative Comparison ===")
    print(f"Regression MSE: {avg_regression_mse:.4f}")
    print(f"CVAE Average MSE: {avg_cvae_mse:.4f}")
    print(f"CVAE Best Sample MSE (Oracle): {avg_best_cvae_mse:.4f}")

    # Visualize comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    models = ['Regression', 'CVAE\n(Average)', 'CVAE\n(Best Sample)']
    mses = [avg_regression_mse, avg_cvae_mse, avg_best_cvae_mse]

    bars = ax.bar(models, mses, color=['blue', 'orange', 'green'])
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Model Comparison: Reconstruction Quality')
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mse in zip(bars, mses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{mse:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('cvae_regression_quantitative_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    return {
        'regression_mse': avg_regression_mse,
        'cvae_avg_mse': avg_cvae_mse,
        'cvae_best_mse': avg_best_cvae_mse
    }

def compare_latent_dimensions_cvae(train_loader, test_loader, device, dimensions=[2, 5, 10, 20, 50]):
    """Compare CVAE performance with different latent dimensions."""
    results = {}

    for dim in dimensions:
        print(f"\nTraining CVAE with latent dimension {dim}...")

        # Create and train model
        model = CVAE(latent_dim=dim).to(device)
        train_losses, test_losses = train_cvae(model, train_loader, test_loader,
                                               epochs=50, device=device)

        # Evaluate reconstruction quality
        model.eval()
        total_mse = 0
        total_samples = 0

        with torch.no_grad():
            for x_cond, y_target, _ in test_loader:
                x_cond = x_cond.to(device)
                y_target = y_target.to(device)
                y_logits, _, _ = model(y_target, x_cond)
                y_recon = torch.sigmoid(y_logits)

                mse = F.mse_loss(y_recon, y_target, reduction='sum')
                total_mse += mse.item()
                total_samples += y_target.size(0)

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
    ax1.set_title('CVAE: Reconstruction Quality vs Latent Dimension')
    ax1.grid(True)

    # Test loss curves
    for dim in dimensions:
        ax2.plot(results[dim]['test_losses'], label=f'dim={dim}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('CVAE: Test Loss During Training')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('cvae_dimension_comparison.png', dpi=150, bbox_inches='tight')
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

    # Load and prepare datasets
    transform = transforms.Compose([transforms.ToTensor()])

    train_mnist = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST('./data', train=False, transform=transform)

    # Create modified datasets
    train_dataset = ModifiedMNISTDataset(train_mnist)
    test_dataset = ModifiedMNISTDataset(test_mnist)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize models
    print("\nInitializing models...")
    # cvae_model = CVAE(y_dim=784, x_dim=28, hidden_dim=400, latent_dim=latent_dim).to(device)
    # regression_model = RegressionBaseline(x_dim=28, hidden_dim=400, y_dim=784).to(device)

    # # Train CVAE
    # print("\nTraining CVAE...")
    # cvae_train_losses, cvae_test_losses = train_cvae(
    #     cvae_model, train_loader, test_loader, epochs=epochs, lr=lr, device=device
    # )

    # # Train Regression baseline
    # print("\nTraining Regression baseline...")
    # reg_train_losses, reg_test_losses = train_regression(
    #     regression_model, train_loader, test_loader, epochs=epochs, lr=lr, device=device
    # )

    # # Save models
    # torch.save(cvae_model.state_dict(), 'cvae_model.pth')
    # torch.save(regression_model.state_dict(), 'regression_model.pth')

    # # Plot training curves
    # plt.figure(figsize=(12, 5))

    # plt.subplot(1, 2, 1)
    # plt.plot(cvae_train_losses, label='CVAE Train')
    # plt.plot(cvae_test_losses, label='CVAE Test')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('CVAE Training Curves')
    # plt.legend()
    # plt.grid(True)

    # plt.subplot(1, 2, 2)
    # plt.plot(reg_train_losses, label='Regression Train')
    # plt.plot(reg_test_losses, label='Regression Test')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Regression Training Curves')
    # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.savefig('cvae_regression_training_curves.png', dpi=150, bbox_inches='tight')
    # plt.show()

    # # Visualizations and Analysis
    # print("\nGenerating visualizations and analysis...")

    # # 1. Compare ambiguous completions
    # visualize_ambiguous_completions(cvae_model, regression_model, test_loader, device)

    # # 2. Analyze diversity of CVAE outputs
    # diversity_scores = analyze_diversity(cvae_model, test_loader, device)

    # # 3. Find and visualize truly ambiguous cases
    # ambiguous_cases = find_ambiguous_cases(cvae_model, test_loader, device)

    # # 4. Quantitative comparison
    # comparison_results = compare_models_quantitatively(
    #     cvae_model, regression_model, test_loader, device
    # )

    # print("\n=== Summary ===")
    # print(f"CVAE successfully trained with {latent_dim}-dimensional latent space")
    # print(f"CVAE produces diverse outputs with average diversity score: {np.mean(diversity_scores):.4f}")
    # print(f"Found {len(ambiguous_cases)} ambiguous central columns mapping to multiple digits")
    # print(
    #     f"CVAE oracle performance shows potential {comparison_results['regression_mse'] / comparison_results['cvae_best_mse']:.2f}x improvement over regression")

    # print("\nExperiment complete! Check generated plots for visual analysis.")

    dimension_results = compare_latent_dimensions_cvae(train_loader, test_loader, device, dimensions=[20])


if __name__ == "__main__":
    main()