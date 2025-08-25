"""
AtomEmbedding configs
"""
atom_embedding_dim = 64 
""" Dimension of the atom embeddings """
num_atom_types = 86
""" Number of unique atom types in the dataset """
bond_embedding_dim = 128 
""" Dimension of the bond embeddings """
num_bond_types = 13
""" Number of unique bond types in the dataset """


"""
Edge Attention configs
"""
edge_attetion_output_negative_slope = 0.1
""" Negative slope for the output of the edge attention feedforward network """
num_heads = 4
""" Number of attention heads """
global_num_heads = 4
""" Number of global attention heads """


"""
GeAT configs
"""
backbone_dropout = 0.2
""" Dropout rate for the GeAT backbone """
neck_dropout = 0.2
""" Dropout rate for the GeAT neck """
head_dropout = 0.2
""" Dropout rate for the GeAT head """
geatnet_hidden_dim = 64
""" Hidden dimension for the egeatnet feedforward network """
geatnet_layers = 2
""" Number of GeAT layers in the GeATNet"""
parallel_between_bondtypes = True
""" Whether to use parallel attention mechanisms for different bond types """


"""
Training configs
"""
batch_size = 64
""" Batch size for training """
context_length = 470
""" Maximum number of atoms in a molecule for padding """