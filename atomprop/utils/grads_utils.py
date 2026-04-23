import torch
import cvxpy as cp
import numpy as np

def find_optimal_weights(vector_list):
    """
    Find weights w such that sum(w_i * g_i) approximates the optimal direction g_opt
    PyTorch compatible version
    
    Parameters:
    vector_list: List[torch.Tensor], each tensor is a vector (1D tensor)
    
    Returns:
    w: torch.Tensor of shape (n,), weights satisfying w_i >= 0, sum(w_i) = 1
    """
    if not isinstance(vector_list, list):
        raise TypeError(f"Expected list of tensors, got {type(vector_list)}")
    
    if len(vector_list) == 0:
        raise ValueError("Input list cannot be empty")
    
    for i, v in enumerate(vector_list):
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"Element {i} is not a torch.Tensor, got {type(v)}")
        if v.dim() != 1:
            raise ValueError(f"Element {i} must be a 1D tensor, got shape {v.shape}")
    
    vectors = torch.stack(vector_list, dim=0)
    vectors_np = vectors.cpu().numpy()
        
    # Normalize to unit vectors
    norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
    vectors_np = vectors_np / (norms + 1e-8)
    
    n, d = vectors_np.shape
    
    # First find g_opt using the original convex formulation
    g = cp.Variable(d)
    t = cp.Variable()
    
    constraints = []
    for i in range(n):
        constraints.append(vectors_np[i] @ g >= t)
    constraints.append(cp.norm(g) <= 1)
    
    prob = cp.Problem(cp.Maximize(t), constraints)
    
    # Try to solve with available solver
    try:
        # Try ECOS first
        prob.solve(solver=cp.ECOS, verbose=False)
    except:
        try:
            # Fallback to SCS
            prob.solve(solver=cp.SCS, max_iters=10000, verbose=False)
        except:
            # Try default solver
            prob.solve(verbose=False)
    
    if g.value is None:
        raise ValueError("Solver failed, no feasible solution found")
    
    g_opt = g.value
    g_norm = np.linalg.norm(g_opt)
    if g_norm > 0:
        g_opt = g_opt / g_norm
    
    # Now solve for weights: minimize ||sum(w_i * g_i) - g_opt||^2
    # This is a convex quadratic problem with linear constraints
    w = cp.Variable(n)
    
    # Weighted combination
    weighted_sum = vectors_np.T @ w  # d-dimensional vector
    
    # Objective: minimize squared distance between weighted sum and g_opt
    objective = cp.Minimize(cp.sum_squares(weighted_sum - g_opt))
    
    # Constraints: non-negative weights that sum to 1
    constraints = [w >= 0, cp.sum(w) == 1]
    
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except:
        try:
            prob.solve(solver=cp.SCS, max_iters=10000, verbose=False)
        except:
            prob.solve(verbose=False)
    
    if w.value is None:
        # Fallback: uniform weights
        w_opt = np.ones(n) / n
    else:
        w_opt = w.value
        # Ensure non-negativity and normalization
        w_opt = np.maximum(w_opt, 0)
        w_sum = np.sum(w_opt)
        if w_sum > 0:
            w_opt = w_opt / w_sum
        else:
            w_opt = np.ones(n) / n
    
    # Convert back to torch tensor
    return torch.from_numpy(w_opt).float()
