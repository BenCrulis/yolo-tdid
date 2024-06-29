import torch


def whitening_matrix(X, eps=1e-5):
    # Compute the covariance matrix
    cov_matrix = torch.cov(X.T)
    
    # Perform eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    
    # Form the whitening matrix
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(eigenvalues + eps))
    W = eigenvectors @ D_inv_sqrt @ eigenvectors.T
    
    return W


def coloring_matrix(target_cov_matrix):
    # Perform eigen decomposition on the target covariance matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(target_cov_matrix)
    eigenvalues = torch.maximum(eigenvalues, torch.zeros_like(eigenvalues)) # avoid negative eigenvalues

    if (eigenvalues == 0.0).sum() > 0:
        print("Warning: some eigenvalues are negative, clamping to zero.")
    
    # Form the coloring matrix
    D_sqrt = torch.diag(torch.sqrt(eigenvalues))
    coloring_matrix = eigenvectors @ D_sqrt @ eigenvectors.T
    
    return coloring_matrix