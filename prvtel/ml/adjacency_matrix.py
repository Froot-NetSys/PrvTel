import numpy as np

def create_adjacency_matrix(bn_structure, feature_names, expanded_feature_names):
    """
    Creates an expanded adjacency matrix from a Bayesian network structure, accounting for GMM-expanded features.
    
    Args:
        bn_structure (list): List of tuples where each tuple contains (child, [parents])
        feature_names (list): List of original feature names
        expanded_feature_names (list): List of feature names after GMM expansion
    
    Returns:
        numpy.ndarray: Expanded adjacency matrix matching GMM dimensions
    """
    n_expanded = len(expanded_feature_names)
    
    # Create mapping of original features to their expanded feature indices
    feature_to_expanded_idx = {}
    for orig_feature in feature_names:
        # Find all expanded features that contain the original feature name
        # Modified pattern matching to handle GMM-expanded features
        expanded_indices = [i for i, name in enumerate(expanded_feature_names) 
                          if orig_feature in name.split('_')[0]]  # Split by underscore to handle GMM suffixes
        feature_to_expanded_idx[orig_feature] = expanded_indices
        
    # Initialize expanded adjacency matrix
    expanded_adj_matrix = np.zeros((n_expanded, n_expanded))
    
    # Fill the expanded adjacency matrix based on the BN structure
    for child, parents in bn_structure:
        if child in feature_to_expanded_idx:
            child_indices = feature_to_expanded_idx[child]
            for parent in parents:
                if parent in feature_to_expanded_idx:
                    parent_indices = feature_to_expanded_idx[parent]
                    # Set connections between all expanded features
                    for ci in child_indices:
                        for pi in parent_indices:
                            expanded_adj_matrix[ci][pi] = 1
    
    return expanded_adj_matrix