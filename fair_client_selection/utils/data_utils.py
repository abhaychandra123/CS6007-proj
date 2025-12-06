"""Data loading and partitioning utilities for federated learning."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


class TabularDataset(Dataset):
    """Dataset for tabular data with sensitive attributes."""
    
    def __init__(self, X, y, sensitive_attr):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.sensitive_attr = torch.LongTensor(sensitive_attr)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.sensitive_attr[idx]


def load_cifar10(data_dir='./data'):
    """Load CIFAR-10 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    
    return trainset, testset


def create_synthetic_adult_data(n_samples=10000, n_features=14, seed=42):
    """
    Create synthetic Adult Census-like dataset with sensitive attribute.
    
    Returns:
        X: Features (n_samples, n_features)
        y: Binary labels (income >50K or not)
        sensitive_attr: Binary sensitive attribute (e.g., gender)
    """
    np.random.seed(seed)
    
    # Generate features
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Generate sensitive attribute (correlated with some features)
    sensitive_attr = (X[:, 0] + 0.5 * np.random.randn(n_samples) > 0).astype(int)
    
    # Generate labels with bias based on sensitive attribute
    logits = (
        X[:, 0] * 0.5 +
        X[:, 1] * 0.3 +
        X[:, 2] * 0.2 +
        sensitive_attr * 0.4 +  # Bias term
        np.random.randn(n_samples) * 0.3
    )
    y = (logits > 0).astype(int)
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, sensitive_attr


def partition_data_dirichlet(
    dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42
) -> List[List[int]]:
    """
    Partition dataset using Dirichlet distribution (non-IID).
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        seed: Random seed
    
    Returns:
        List of index lists for each client
    """
    np.random.seed(seed)
    
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]
    
    # For each class, distribute samples to clients using Dirichlet
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        # Sample from Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array([p * (len(idx_j) < len(dataset) / num_clients) 
                               for p, idx_j in zip(proportions, client_indices)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        # Split indices according to proportions
        idx_k_split = np.split(idx_k, proportions)
        for idx_j, idx in zip(client_indices, idx_k_split):
            idx_j.extend(idx.tolist())
    
    return client_indices


def partition_adult_data_by_state(
    X, y, sensitive_attr,
    num_clients: int = 50,
    seed: int = 42
) -> Tuple[List[TabularDataset], List[TabularDataset]]:
    """
    Partition Adult-like data to simulate state-level heterogeneity.
    
    Each client (state) has different data distributions and fairness properties.
    """
    np.random.seed(seed)
    
    # Split into train/test
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive_attr, test_size=0.2, random_state=seed
    )
    
    # Create heterogeneous distributions for clients
    n_train = len(X_train)
    n_test = len(X_test)
    
    # Use Dirichlet to create non-uniform partitions
    proportions = np.random.dirichlet(np.repeat(0.5, num_clients))
    train_splits = (np.cumsum(proportions) * n_train).astype(int)[:-1]
    test_splits = (np.cumsum(proportions) * n_test).astype(int)[:-1]
    
    train_indices = np.split(np.arange(n_train), train_splits)
    test_indices = np.split(np.arange(n_test), test_splits)
    
    client_train_datasets = []
    client_test_datasets = []
    
    for train_idx, test_idx in zip(train_indices, test_indices):
        # Add bias to some clients to create fairness heterogeneity
        train_dataset = TabularDataset(
            X_train[train_idx],
            y_train[train_idx],
            s_train[train_idx]
        )
        test_dataset = TabularDataset(
            X_test[test_idx],
            y_test[test_idx],
            s_test[test_idx]
        )
        
        client_train_datasets.append(train_dataset)
        client_test_datasets.append(test_dataset)
    
    return client_train_datasets, client_test_datasets


def add_sensitive_attribute_to_cifar10(dataset, seed=42):
    """
    Add synthetic sensitive attribute to CIFAR-10 dataset.
    
    Sensitive attribute is correlated with certain classes to create bias.
    """
    np.random.seed(seed)
    
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array(dataset.labels)
    
    # Create biased sensitive attribute
    # Classes 0-4 are more likely to be group 0, classes 5-9 more likely group 1
    sensitive_attr = np.zeros(len(labels), dtype=int)
    
    for i in range(len(labels)):
        if labels[i] < 5:
            sensitive_attr[i] = int(np.random.rand() < 0.7)  # 70% group 0
        else:
            sensitive_attr[i] = int(np.random.rand() < 0.3)  # 30% group 0
    
    return sensitive_attr


class CIFAR10WithSensitive(Dataset):
    """CIFAR-10 dataset with sensitive attribute."""
    
    def __init__(self, cifar_dataset, sensitive_attr):
        self.cifar_dataset = cifar_dataset
        self.sensitive_attr = torch.LongTensor(sensitive_attr)
        
    def __len__(self):
        return len(self.cifar_dataset)
    
    def __getitem__(self, idx):
        img, label = self.cifar_dataset[idx]
        return img, label, self.sensitive_attr[idx]


def get_federated_cifar10(
    num_clients: int = 10,
    alpha: float = 0.5,
    data_dir: str = './data',
    seed: int = 42
) -> Tuple[List[Dataset], List[Dataset], Dataset]:
    """
    Get federated CIFAR-10 with sensitive attributes.
    
    Returns:
        client_train_datasets: List of training datasets for each client
        client_test_datasets: List of test datasets for each client
        global_test: Global test dataset
    """
    trainset, testset = load_cifar10(data_dir)
    
    # Add sensitive attributes
    train_sensitive = add_sensitive_attribute_to_cifar10(trainset, seed)
    test_sensitive = add_sensitive_attribute_to_cifar10(testset, seed + 1)
    
    # Wrap datasets
    trainset = CIFAR10WithSensitive(trainset, train_sensitive)
    testset = CIFAR10WithSensitive(testset, test_sensitive)
    
    # Partition training data
    client_indices = partition_data_dirichlet(trainset, num_clients, alpha, seed)
    
    client_train_datasets = []
    for indices in client_indices:
        client_train_datasets.append(Subset(trainset, indices))
    
    # For test data, also partition (client-specific test)
    test_indices = partition_data_dirichlet(testset, num_clients, alpha, seed + 100)
    client_test_datasets = []
    for indices in test_indices:
        client_test_datasets.append(Subset(testset, indices))
    
    # Global test set
    global_test = testset
    
    return client_train_datasets, client_test_datasets, global_test


def get_federated_adult(
    num_clients: int = 50,
    n_samples: int = 10000,
    seed: int = 42
) -> Tuple[List[TabularDataset], List[TabularDataset], TabularDataset]:
    """
    Get federated Adult Census-like dataset.
    
    Returns:
        client_train_datasets: List of training datasets
        client_test_datasets: List of test datasets
        global_test: Global test dataset
    """
    # Create synthetic data
    X, y, sensitive_attr = create_synthetic_adult_data(n_samples, seed=seed)
    
    # Partition by state
    client_train, client_test = partition_adult_data_by_state(
        X, y, sensitive_attr, num_clients, seed
    )
    
    # Create separate global test set
    X_global, y_global, s_global = create_synthetic_adult_data(2000, seed=seed + 1000)
    global_test = TabularDataset(X_global, y_global, s_global)
    
    return client_train, client_test, global_test
