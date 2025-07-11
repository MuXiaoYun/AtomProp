"""
AtomEmbedding configs
"""
atom_embedding_dim = 64 
""" Dimension of the atom embeddings """
num_atom_types = 54
""" Number of unique atom types in the dataset """
num_bond_types = 4
""" Number of unique bond types in the dataset """


"""
Edge Attention configs
"""
edge_attention_hidden_dim = 64
""" Hidden dimension for the edge attention feedforward network """
edge_attetion_output_negative_slope = 0.1
""" Negative slope for the output of the edge attention feedforward network """


"""
GeAT configs
"""
geatnet_hidden_dim = 64
""" Hidden dimension for the egeatnet feedforward network """
geatnet_layers = 2
""" Number of GeAT layers in the GeATNet"""