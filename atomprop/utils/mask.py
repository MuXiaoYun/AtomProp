"""
Module for graph masking.
"""

import torch
import random

class MolGraphMask:
    """
    Class for masking atoms in mol graphs.
    """
    @staticmethod
    def select_mask_indices(atom_list, mask_ratio = 0.1):
        """
        Select random atom indices to mask.
        :param atom_list: List of atom embeddings.
        :param mask_ratio: Ratio of atoms to mask.
        :return: List of indices to mask.
        """
        atom_indice_dict = {}
        mask_indices = []
        
        for idx, atom in enumerate(atom_list):
            # Each atom has multiple indices, store them in a list in the dict
            if atom not in atom_indice_dict:
                atom_indice_dict[atom] = [idx]
                continue
            atom_indice_dict[atom].append(idx)

        sample_num = max(1, int(len(atom_list) * mask_ratio / len(atom_indice_dict)))

        for key in atom_indice_dict:
            # sample sample_num indices from each atom's indices. accept duplicates
            sampled_indices = random.sample(atom_indice_dict[key], min(sample_num, len(atom_indice_dict[key])))
            mask_indices.extend(sampled_indices)
        
        return mask_indices

    @staticmethod
    def mask_atoms(embed_list, mask_indices, mask_token, modify = True):
        """
        Mask selected atoms in the embedding list.
        :param embed_list: List of atom embeddings.
        :param mask_indices: List of indices to mask.
        :param mask_token: Embedding to use for masking.
        :param modify: Whether to modify the original embed_list or return a new one.
        :return: New list of embeddings with masked atoms.
        """
        if modify == False:
            masked_embed_list = embed_list.clone()
            for idx in mask_indices:
                masked_embed_list[idx] = mask_token
            return masked_embed_list

        else:
            for idx in mask_indices:
                embed_list[idx] = mask_token
            return embed_list
