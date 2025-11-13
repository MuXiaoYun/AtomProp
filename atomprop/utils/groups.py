"""
Module for generating A-B-C triplet or A-B-C-D quadruplet atom groups from molecules.
These groups can be used for various sub-structure level prediction tasks.
"""

from rdkit import Chem
import torch
from torch_geometric.data import Data, Batch

class TripletGroup:
    """
    Generate A-B-C triplet atom groups from some molecules.
    1st atom -- center atom -- 3rd atom
    1-c-3 and 3-c-1 are treated as the same group.
    """

    @staticmethod
    def batch_generate(edge_index):
        """
        Generate triplet groups for a batch of molecules from edge indices of the mol graph.
        View the whole batch as one molecule and calculate. 
        1-c-3 and 3-c-1 are treated as the same group.
        Return in pytorch tensor, shape should be (num_groups, 3).
        
        Args:
            edge_index (torch.Tensor): Edge index tensor of shape (2, num_edges)
            
        Returns:
            torch.Tensor: Triplet groups tensor of shape (num_groups, 3)
                        Each row represents a triplet (atom1, center_atom, atom2)
        """
        # Convert to set for faster lookup and to remove duplicates
        edge_set = set()
        num_edges = edge_index.size(1)
        
        # Store edges in both directions to handle undirected graph
        for i in range(num_edges):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_set.add((src, dst))
            edge_set.add((dst, src))  # Add reverse edge for undirected graph
        
        triplets = set()
        
        # Find all triplets: for each center atom, find all pairs of its neighbors
        # Build adjacency list
        adj_list = {}
        for src, dst in edge_set:
            if src not in adj_list:
                adj_list[src] = set()
            adj_list[src].add(dst)
        
        # Generate triplets: for each center atom, combine all pairs of its neighbors
        for center_atom in adj_list:
            neighbors = list(adj_list[center_atom])
            num_neighbors = len(neighbors)
            
            # Generate all unique pairs of neighbors
            for i in range(num_neighbors):
                for j in range(i + 1, num_neighbors):
                    atom1, atom2 = neighbors[i], neighbors[j]
                    
                    # Create canonical triplet representation to avoid duplicates
                    # Sort the two end atoms to treat (a1, center, a2) and (a2, center, a1) as same
                    if atom1 < atom2:
                        triplet = (atom1, center_atom, atom2)
                    else:
                        triplet = (atom2, center_atom, atom1)
                    
                    triplets.add(triplet)
        
        # Convert to tensor
        if triplets:
            triplet_tensor = torch.tensor(list(triplets), dtype=torch.long)
        else:
            # Return empty tensor with correct shape if no triplets found
            triplet_tensor = torch.empty((0, 3), dtype=torch.long)
        
        return triplet_tensor

class QuadrupletGroup:
    """
    Generate A-B-C-D quadruplet atom groups from some molecules.
    1st atom -- center atom1 -- center atom2 -- 4th atom
    """

    @staticmethod
    def batch_generate(edge_index):
        """
        Generate quadruplet groups for a batch of molecules from edge indices of the mol graph.
        View the whole batch as one molecule and calculate. 
        Return in pytorch tensor, shape should be (num_groups, 4).
        
        Args:
            edge_index (torch.Tensor): Edge index tensor of shape (2, num_edges)
        Returns:
            torch.Tensor: Quadruplet groups tensor of shape (num_groups, 4)
                        Each row represents a quadruplet (atom1, center_atom1, center_atom2, atom4)
        """
        # Convert to set for faster lookup and to remove duplicates
        edge_set = set()
        num_edges = edge_index.size(1)
        
        # Store edges in both directions to handle undirected graph
        for i in range(num_edges):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_set.add((src, dst))
            edge_set.add((dst, src))  # Add reverse edge for undirected graph
        
        quadruplets = set()
        
        # Build adjacency list
        adj_list = {}
        for src, dst in edge_set:
            if src not in adj_list:
                adj_list[src] = set()
            adj_list[src].add(dst)
        
        # Generate quadruplets: for each pair of connected center atoms, find neighbors
        for center_atom1 in adj_list:
            for center_atom2 in adj_list[center_atom1]:
                if center_atom1 >= center_atom2:
                    continue  # Avoid duplicate pairs
                
                neighbors1 = adj_list[center_atom1] - {center_atom2}
                neighbors2 = adj_list[center_atom2] - {center_atom1}
                
                for atom1 in neighbors1:
                    for atom4 in neighbors2:
                        quadruplet = (atom1, center_atom1, center_atom2, atom4)
                        quadruplets.add(quadruplet)
        
        # Convert to tensor
        if quadruplets:
            quadruplet_tensor = torch.tensor(list(quadruplets), dtype=torch.long)
        else:
            # Return empty tensor with correct shape if no quadruplets found
            quadruplet_tensor = torch.empty((0, 4), dtype=torch.long)
        
        return quadruplet_tensor
