import torch


def whitening_matrix(X):
    # Center the data
    X_centered = X - torch.mean(X, axis=0)
    
    # Compute the covariance matrix
    cov_matrix = torch.cov(X_centered.T)
    
    # Perform eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    
    # Form the whitening matrix
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(eigenvalues))
    W = eigenvectors @ D_inv_sqrt @ eigenvectors.T
    
    return W


def coloring_matrix(target_cov_matrix):
    # Perform eigen decomposition on the target covariance matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(target_cov_matrix)
    
    # Form the coloring matrix
    D_sqrt = torch.diag(torch.sqrt(eigenvalues))
    coloring_matrix = eigenvectors @ D_sqrt @ eigenvectors.T
    
    return coloring_matrix