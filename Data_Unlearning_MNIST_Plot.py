#!/usr/bin/env python
"""
Machine Unlearning via Information Regularization on MNIST

This script trains several MNIST classifiers:
  1. An original (non-regularized) model on the full training set.
  2. A model re-trained from scratch using only the "remain" data.
  3. A model fine-tuned (continued training) from the original model.
  4. Several unlearning experiments in which we modify the loss to include a mutual
     information (MI) penalty. For each unlearning experiment (with different gamma/epoch settings),
     we run 10 repeated trials and then plot the mean & std of MI and various accuracies.

The final output is four ICML-style plots saved as PNG files.
"""

import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt

# --- Set device ---
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# --- Define the MNIST Classifier ---
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- Basic Training and Evaluation functions ---
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Accuracy: {acc:.2f}%")
    return correct / total

# --- Unlearning functions ---
def create_train_labels(remain_images, unlearn_images):
    """
    Create the combined training set for the mutual information loss.
    We label (remain + unlearn) images as z=0 and remain images as z=1.
    """
    X_0 = torch.cat([remain_images, unlearn_images], dim=0)  # shape: [2B, ...]
    Z_0 = torch.zeros(len(X_0), device=X_0.device)           # label 0 for these images
    X_1 = remain_images                                      # shape: [B, ...]
    Z_1 = torch.ones(len(X_1), device=X_1.device)            # label 1 for these images
    return X_0, Z_0, X_1, Z_1

def compute_mutual_information_multiclass(predicted_outputs, labels):
    """
    Compute an approximation to the mutual information I(Y;Z) for a multi-class output.
    predicted_outputs: logits (N x C)
    labels: binary labels (N,)
    """
    eps = 1e-10
    probs = torch.softmax(predicted_outputs, dim=1).clamp(eps, 1.0 - eps)
    N, C = probs.shape

    p_z1 = labels.float().mean()
    p_z0 = 1.0 - p_z1

    joint_p_yz1 = (probs * labels.unsqueeze(1)).sum(dim=0) / N
    joint_p_yz0 = (probs * (1 - labels).unsqueeze(1)).sum(dim=0) / N

    p_y = joint_p_yz1 + joint_p_yz0

    term_yz1 = joint_p_yz1 * torch.log(joint_p_yz1 / (p_y * p_z1 + eps) + eps)
    term_yz0 = joint_p_yz0 * torch.log(joint_p_yz0 / (p_y * p_z0 + eps) + eps)
    mi = term_yz1.sum() + term_yz0.sum()
    return mi

def train_unlearn_model(model, train_unlearn_loader, train_remain_loader, criterion, optimizer, gamma=1.0, epochs=10):
    """
    Trains the model with both the standard CE loss on the remaining data and an MI penalty.
    Returns histories of CE, MI, accuracies, and total loss.
    """
    model.train()

    mi_history = []
    acc_unlearn_history = []
    acc_remain_history = []
    acc_test_history = []
    ce_history = []
    total_loss_history = []

    for epoch in range(epochs):
        epoch_total_loss = 0.0
        epoch_ce = 0.0
        epoch_mi = 0.0
        n_batches = 0

        for (images_remain, digits_remain), (images_unlearn, digits_unlearn) in zip(train_remain_loader, train_unlearn_loader):
            images_remain = images_remain.to(device)
            digits_remain = digits_remain.to(device)
            images_unlearn = images_unlearn.to(device)
            digits_unlearn = digits_unlearn.to(device)
            optimizer.zero_grad()

            outputs_remain = model(images_remain)
            ce_loss = criterion(outputs_remain, digits_remain)

            X_0, Z_0, X_1, Z_1 = create_train_labels(images_remain, images_unlearn)
            X_train = torch.cat((X_0, X_1), dim=0)
            Z_train = torch.cat((Z_0, Z_1), dim=0)
            idx = torch.randperm(len(X_train), device=device)
            X_train_shuffled = X_train[idx]
            Z_train_shuffled = Z_train[idx]

            Y_train_shuffled = model(X_train_shuffled)
            mi_val = compute_mutual_information_multiclass(Y_train_shuffled, Z_train_shuffled)

            total_loss = ce_loss + gamma * mi_val
            total_loss.backward()
            optimizer.step()

            epoch_ce += ce_loss.item()
            epoch_mi += mi_val.item()
            epoch_total_loss += total_loss.item()
            n_batches += 1

        avg_ce = epoch_ce / n_batches
        avg_mi = epoch_mi / n_batches
        avg_loss = epoch_total_loss / n_batches

        avg_acc_unlearn = evaluate_model(model, train_unlearn_loader)
        avg_acc_remain = evaluate_model(model, train_remain_loader)
        avg_acc_test = evaluate_model(model, test_loader)

        ce_history.append(avg_ce)
        mi_history.append(avg_mi)
        acc_unlearn_history.append(avg_acc_unlearn)
        acc_remain_history.append(avg_acc_remain)
        acc_test_history.append(avg_acc_test)
        total_loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs} | CE: {avg_ce:.4f}, MI: {avg_mi:.4f}, Total Loss: {avg_loss:.4f}")

    return ce_history, mi_history, acc_unlearn_history, acc_remain_history, acc_test_history, total_loss_history

# --- Plotting function ---
def plot_icml_style(mi_history_list, acc_unlearn_history_list, acc_remain_history_list, acc_test_history_list, epochs, gamma, filename="icml_plot.png"):
    """
    Creates an ICML-style plot with shaded standard deviations.
    The title now includes the gamma value.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    # Removed the extra plt.figure() call to avoid creating an empty figure.

    colors = {"blue": "#0072B2", "red": "#D55E00", "green": "#009E73", "purple": "#CC79A7"}

    def compute_mean_std(history_list):
        arr = np.array(history_list)
        return arr.mean(axis=0), arr.std(axis=0)

    mi_mean, mi_std = compute_mean_std(mi_history_list)
    acc_unlearn_mean, acc_unlearn_std = compute_mean_std(acc_unlearn_history_list)
    acc_remain_mean, acc_remain_std = compute_mean_std(acc_remain_history_list)
    acc_test_mean, acc_test_std = compute_mean_std(acc_test_history_list)

    epochs_range = range(1, epochs + 1)

    fig, ax1 = plt.subplots(figsize=(7, 5))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
        "axes.linewidth": 1,
    })

    ax1.set_xlabel("Epochs", fontsize=16)
    ax1.set_ylabel("Mutual Information", color=colors["blue"], fontsize=16)
    mi_plot, = ax1.plot(epochs_range, mi_mean, label="Mutual Information", color=colors["blue"], linewidth=2)
    ax1.fill_between(epochs_range, mi_mean - mi_std, mi_mean + mi_std, color=colors["blue"], alpha=0.2)
    ax1.tick_params(axis="y", labelcolor=colors["blue"], width=1.2)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", fontsize=16)
    acc_unlearn_plot, = ax2.plot(epochs_range, acc_unlearn_mean, label="Unlearn Accuracy", color=colors["red"], linestyle="dashed", linewidth=2)
    ax2.fill_between(epochs_range, acc_unlearn_mean - acc_unlearn_std, acc_unlearn_mean + acc_unlearn_std, color=colors["red"], alpha=0.2)
    acc_remain_plot, = ax2.plot(epochs_range, acc_remain_mean, label="Remain Accuracy", color=colors["green"], linestyle="dashed", linewidth=2)
    ax2.fill_between(epochs_range, acc_remain_mean - acc_remain_std, acc_remain_mean + acc_remain_std, color=colors["green"], alpha=0.2)
    acc_test_plot, = ax2.plot(epochs_range, acc_test_mean, label="Test Accuracy", color=colors["purple"], linestyle="dashed", linewidth=2)
    ax2.fill_between(epochs_range, acc_test_mean - acc_test_std, acc_test_mean + acc_test_std, color=colors["purple"], alpha=0.2)
    ax2.tick_params(axis="y", width=1.2)

    lines = [mi_plot, acc_unlearn_plot, acc_remain_plot, acc_test_plot]
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc="upper right", frameon=True, edgecolor='black')

    fig.tight_layout()
    plt.title(f"Mutual Information & Accuracy vs. Epochs (Gamma = {gamma})", fontsize=14, fontweight="bold")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {filename}")
    plt.show()
    plt.close()

# --- Main function ---
if __name__ == "__main__":
    # --- Data Preparation ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create the "unlearn" and "remain" datasets.
    # Here we use all images with label==3 for unlearning (selecting 75% of them randomly)
    label_indices = [i for i, (_, label) in enumerate(train_dataset) if label == 3]
    selected_unlearn_indices = random.sample(label_indices, int(len(label_indices) * 0.75))
    remaining_indices = [i for i in range(len(train_dataset)) if i not in selected_unlearn_indices]

    unlearn_dataset = Subset(train_dataset, selected_unlearn_indices)
    remaining_dataset = Subset(train_dataset, remaining_indices)

    train_unlearn_loader = DataLoader(unlearn_dataset, batch_size=16, shuffle=True)
    train_remain_loader = DataLoader(remaining_dataset, batch_size=64, shuffle=True)

    # --- Original Training Experiments ---
    print("\n==== Training Original (Non-Regularized) Model ====")
    model_original = MNISTClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_original.parameters(), lr=0.001)
    train_model(model_original, train_loader, criterion, optimizer, epochs=100)
    print("Evaluating Original Model:")
    evaluate_model(model_original, train_remain_loader)
    evaluate_model(model_original, train_unlearn_loader)
    evaluate_model(model_original, test_loader)

    print("\n==== Re-Training from Scratch on 'Remain' Data ====")
    model_retrain = MNISTClassifier().to(device)
    optimizer = optim.Adam(model_retrain.parameters(), lr=0.001)
    train_model(model_retrain, train_remain_loader, criterion, optimizer, epochs=100)
    print("Evaluating Retrained Model:")
    evaluate_model(model_retrain, train_remain_loader)
    evaluate_model(model_retrain, train_unlearn_loader)
    evaluate_model(model_retrain, test_loader)

    print("\n==== Fine-Tuning (Re-Training from Learned Outcome) on 'Remain' Data ====")
    model_retrain_continue = deepcopy(model_original)
    optimizer = optim.Adam(model_retrain_continue.parameters(), lr=0.001)
    train_model(model_retrain_continue, train_remain_loader, criterion, optimizer, epochs=100)
    print("Evaluating Fine-Tuned Model:")
    evaluate_model(model_retrain_continue, train_remain_loader)
    evaluate_model(model_retrain_continue, train_unlearn_loader)
    evaluate_model(model_retrain_continue, test_loader)

    # --- Unlearning Experiments (Repeated Trials) ---
    model_original.eval()  # switch to eval mode
    print("\n==== Starting Unlearning Experiments ====")
    
    # Define experiments as tuples: (gamma, epochs, filename)
    experiments = [
        (1.5, 20, "icml_plot_exp1.png"),
        (3.5, 5,  "icml_plot_exp2.png"),
        (3.5, 20, "icml_plot_exp3.png"),
        (5,   2,  "icml_plot_exp4.png")
    ]
    
    for idx, (gamma, exp_epochs, filename) in enumerate(experiments, start=1):
        print(f"\n--- Unlearning Experiment {idx}: gamma={gamma}, epochs={exp_epochs} ---")
        mi_history_list = []
        acc_unlearn_history_list = []
        acc_remain_history_list = []
        acc_test_history_list = []
        for trial in range(10):
            print(f"  Trial {trial+1}/10")
            model_unlearn = deepcopy(model_original)
            optimizer = optim.Adam(model_unlearn.parameters(), lr=0.001)
            ce_history, mi_history, acc_unlearn_history, acc_remain_history, acc_test_history, total_loss_history = train_unlearn_model(
                model_unlearn, train_unlearn_loader, train_remain_loader, criterion, optimizer, gamma=gamma, epochs=exp_epochs
            )
            mi_history_list.append(mi_history)
            acc_unlearn_history_list.append(acc_unlearn_history)
            acc_remain_history_list.append(acc_remain_history)
            acc_test_history_list.append(acc_test_history)
        plot_icml_style(mi_history_list, acc_unlearn_history_list, acc_remain_history_list, acc_test_history_list,
                        epochs=exp_epochs, gamma=gamma, filename=filename)