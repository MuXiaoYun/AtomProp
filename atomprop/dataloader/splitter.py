import numpy as np
from typing import List, Tuple, Optional, Generator
from deepchem.data import Dataset
from deepchem.splits import Splitter
import random
import logging

logger = logging.getLogger(__name__)

class ScaffoldSplitter(Splitter):
    """
    Class for doing splits based on the scaffold of small molecules.
    Implentation reference:
    https://github.com/tencent-ailab/grover/blob/main/grover/util/utils.py
    """
    def __init__(self):
        super().__init__()

    def split(
        self,
        dataset: Dataset,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: Optional[int] = None,
        balance: bool = True,
        log_every_n: Optional[int] = 1000
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Splits internal compounds into train/validation/test by scaffold.
        
        This method is overridden to maintain compatibility with the parent class,
        but it ignores the parameters and uses the fold and frac_test parameters
        from initialization.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        frac_train: float, (default 0.8)
        frac_valid: float, (default 0.1)
        frac_test: float, (default 0.1)
        seed: int, optional (default None)
            Random seed to use (only in use if balance is True).
        balance: bool,
            Decide if use balanced scaffold split.
        log_every_n: int, optional (default 1000)
            Controls the logger by dictating how often logger outputs
            will be produced.

        Returns
        -------
        Tuple[List[int], List[int], List[int]]
            A tuple of train indices, valid indices, and test indices.
        """
        scaffold_sets = self._generate_scaffolds(dataset=dataset, log_every_n=log_every_n)
        
        train_size = frac_train * len(dataset)
        val_size = frac_valid * len(dataset)
        test_size = frac_test * len(dataset)
        
        if balance:
            big_sets = []
            small_sets = []
            for scaffold_set in scaffold_sets:
                if len(scaffold_set) > val_size / 2 or len(scaffold_set) > test_size / 2:
                    big_sets.append(scaffold_set)
                else:
                    small_sets.append(scaffold_set)
            random.seed(seed)
            random.shuffle(big_sets)
            random.shuffle(small_sets)
            sorted_sets = big_sets + small_sets
        else:
            sorted_sets = sorted(scaffold_sets, key=len, reverse=True)
            
        train_inds = []
        val_inds = []
        test_inds = []
        for scaffold_set in sorted_sets:
            if len(train_inds) + len(scaffold_set) <= train_size:
                train_inds.extend(scaffold_set)
            elif len(val_inds) + len(scaffold_set) <= val_size:
                val_inds.extend(scaffold_set)
            else:
                test_inds.extend(scaffold_set)
        return train_inds, val_inds, test_inds

    def _generate_scaffolds(self,
                           dataset: Dataset,
                           log_every_n: int = 1000) -> List[List[int]]:
        """Returns all scaffolds from the dataset (copied from ScaffoldSplitter).

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        log_every_n: int, optional (default 1000)
            Controls the logger by dictating how often logger outputs
            will be produced.

        Returns
        -------
        scaffold_sets: List[List[int]]
            List of indices of each scaffold in the dataset, sorted by size.
        """
        scaffolds = {}
        data_len = len(dataset)

        logger.info("About to generate scaffolds")
        
        # Try to import RDKit for scaffold generation
        try:
            from deepchem.splits.splitters import _generate_scaffold
        except ImportError:
            raise ImportError("This splitter requires RDKit to be installed.")
        
        for ind, smiles in enumerate(dataset.ids):
            if ind % log_every_n == 0:
                logger.info("Generating scaffold %d/%d" % (ind, data_len))
            scaffold = _generate_scaffold(smiles)
            if scaffold is not None:
                if scaffold not in scaffolds:
                    scaffolds[scaffold] = [ind]
                else:
                    scaffolds[scaffold].append(ind)

        # Sort from largest to smallest scaffold sets
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set
            for (scaffold,
                 scaffold_set) in sorted(scaffolds.items(),
                                         key=lambda x: (len(x[1]), x[1][0]),
                                         reverse=True)
        ]
        return scaffold_sets