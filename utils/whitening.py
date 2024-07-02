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


def get_whitening_and_coloring_matrices(args, device="cpu"):
    if args.wc is None or not args.wc:
        return None
    
    from utils.whitening import whitening_matrix, coloring_matrix

    print("Loading whitening and coloring data")
    print(f"Opening {args.whitening_data}")
    with open(args.whitening_data, "rb") as f:
        whitening_data = torch.load(f, map_location=device).float()
    
    print(f"Opening {args.coloring_data}")
    with open(args.coloring_data, "rb") as f:
        coloring_data = torch.load(f, map_location=device).float()

    whitening_data_n = whitening_data / whitening_data.norm(p=2, dim=-1, keepdim=True)
    coloring_data_n = coloring_data / coloring_data.norm(p=2, dim=-1, keepdim=True)
    
    print("Computing whitening and coloring matrices")
    W_whitening = whitening_matrix(whitening_data_n, eps=1e-4) # apprently, epsilon is very important

    target_cov_matrix = torch.cov(coloring_data_n.T)
    W_coloring = coloring_matrix(target_cov_matrix)

    w_bias = whitening_data_n.mean(axis=0)  # assume zero bias in the embeddings in order to preserve angles
    c_bias = coloring_data_n.mean(axis=0)

    # temporary test
    # W_whitening = torch.eye(W_whitening.shape[0], device=device)
    # W_coloring = torch.eye(W_coloring.shape[0], device=device)

    return (W_whitening, W_coloring, w_bias, c_bias)
