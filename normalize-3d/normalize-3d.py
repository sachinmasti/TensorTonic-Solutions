import numpy as np

def normalize_3d(vectors):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    vectors = np.array(vectors, dtype=float)
    
    # Handle single vector case
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return vectors if norm == 0 else vectors / norm
    
    # Handle multiple vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    return vectors / norms