import numpy as np
from typing import List, Tuple, Optional, Generator
from deepchem.data import Dataset
from deepchem.splits import Splitter
import logging

logger = logging.getLogger(__name__)


class ScaffoldKFoldSplitter(Splitter):
    """Class for doing K-Fold splits based on the scaffold of small molecules.

    This splitter combines scaffold-based splitting with K-Fold cross-validation:
    1. First, split out a test set based on scaffold (using frac_test)
    2. Then, split the remaining data into K folds for cross-validation

    Notes
    -----
    - When a SMILES representation of a molecule is invalid, the splitter skips processing
      the datapoint i.e it will not include the molecule in any splits.
    """

    def __init__(self, fold: int = 5, frac_test: float = 0.2):
        """
        Initialize the ScaffoldKFoldSplitter.

        Parameters
        ----------
        fold : int, optional (default 5)
            Number of folds for cross-validation
        frac_test : float, optional (default 0.2)
            Fraction of data to be used for the test set
        """
        super().__init__()
        self.fold = fold
        self.frac_test = frac_test
        
        # Validate parameters
        if not isinstance(fold, int) or fold < 2:
            raise ValueError("fold must be an integer >= 2")
        if not (0 < frac_test < 1):
            raise ValueError("frac_test must be between 0 and 1")

    def split(
        self,
        dataset: Dataset,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: Optional[int] = None,
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
        frac_train: float, optional (default 0.8)
            Ignored in this implementation.
        frac_valid: float, optional (default 0.1)
            Ignored in this implementation.
        frac_test: float, optional (default 0.1)
            Ignored in this implementation (uses self.frac_test).
        seed: int, optional (default None)
            Random seed to use (ignored, as scaffold splitting is deterministic).
        log_every_n: int, optional (default 1000)
            Controls the logger by dictating how often logger outputs
            will be produced.

        Returns
        -------
        Tuple[List[int], List[int], List[int]]
            A tuple of train indices, valid indices, and test indices.
            Returns indices for the first fold.
        """
        # Get the first fold
        kfold_gen = self.k_fold_split(dataset, seed, log_every_n)
        train_inds, valid_inds, test_inds = next(kfold_gen)
        
        return train_inds, valid_inds, test_inds

    def k_fold_split(
        self,
        dataset: Dataset,
        seed: Optional[int] = None,
        log_every_n: Optional[int] = 1000
    ) -> Generator[Tuple[List[int], List[int], List[int]], None, None]:
        """
        Generator that yields train/valid/test indices for each fold.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        seed: int, optional (default None)
            Random seed to use (ignored, as scaffold splitting is deterministic).
        log_every_n: int, optional (default 1000)
            Controls the logger by dictating how often logger outputs
            will be produced.

        Yields
        ------
        Tuple[List[int], List[int], List[int]]
            For each fold, yields (train_indices, valid_indices, test_indices)
            test_indices are the same for all folds
        """
        # Generate scaffold sets
        scaffold_sets = self._generate_scaffolds(dataset, log_every_n)
        print(f"[Scaffold Splitter] {len(scaffold_sets)} scaffold types detected!")
        if len(scaffold_sets) < self.fold + 1: # test set and {fold} folds
            raise ValueError(f"[Scaffold Splitter] There're only {len(scaffold_sets)} scaffold types for {self.fold} folds! Max supported fold number for this dataset is {len(scaffold_sets)-1}")
        
        # First, separate test set based on frac_test
        test_cutoff = self.frac_test * len(dataset)
        train_val_sets = []
        test_inds: List[int] = []
        
        logger.info("Separating test set based on scaffold")
        for scaffold_set in scaffold_sets:
            if len(test_inds) + len(scaffold_set) <= test_cutoff:
                test_inds += scaffold_set
            else:
                train_val_sets.append(scaffold_set)
        
        logger.info(f"Test set size: {len(test_inds)}")
        logger.info(f"Train+Validation sets: {len(train_val_sets)} scaffold groups")
        
        # Flatten scaffold groups for train/validation
        # But we need to keep track of which indices belong to which scaffold group
        # for proper K-Fold splitting
        scaffold_groups = []
        for scaffold_set in train_val_sets:
            scaffold_groups.append(scaffold_set)
        
        # Calculate total size of train+validation data
        total_train_val = sum(len(group) for group in scaffold_groups)
        logger.info(f"Total train+validation data points: {total_train_val}")
        
        # Create K folds from scaffold groups
        # We need to distribute scaffold groups into K folds
        folds = [[] for _ in range(self.fold)]
        
        # Sort scaffold groups by size (largest first) for better balance
        scaffold_groups.sort(key=len, reverse=True)
        
        # Simple greedy algorithm to assign scaffold groups to folds
        for scaffold_group in scaffold_groups:
            # Find fold with minimum current size
            fold_sizes = [len(fold) for fold in folds]
            min_fold_idx = np.argmin(fold_sizes)
            folds[min_fold_idx].extend(scaffold_group)
        
        # Log fold sizes
        for i, fold in enumerate(folds):
            logger.info(f"Fold {i+1} size: {len(fold)}")
        
        # Generate each fold
        for i in range(self.fold):
            # Current fold is validation set
            valid_inds = folds[i]
            
            # All other folds are training set
            train_inds = []
            for j in range(self.fold):
                if j != i:
                    train_inds.extend(folds[j])
            
            logger.info(f"Fold {i+1}: Train={len(train_inds)}, Valid={len(valid_inds)}, Test={len(test_inds)}")
            yield train_inds, valid_inds, test_inds

    def train_valid_test_split(
        self,
        dataset: Dataset,
        seed: Optional[int] = None,
        log_every_n: Optional[int] = 1000
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Convenience method to get the first fold as train/valid/test datasets.

        Parameters
        ----------
        dataset: Dataset
            Dataset to be split.
        seed: int, optional (default None)
            Random seed to use.
        log_every_n: int, optional (default 1000)
            Controls the logger by dictating how often logger outputs
            will be produced.

        Returns
        -------
        Tuple[Dataset, Dataset, Dataset]
            Train, valid, and test datasets for the first fold.
        """
        # Get generator
        kfold_gen = self.k_fold_split(dataset, seed, log_every_n)
        
        # Get first fold
        train_inds, valid_inds, test_inds = next(kfold_gen)
        
        # Create subsets
        train_dataset = dataset.select(train_inds)
        valid_dataset = dataset.select(valid_inds)
        test_dataset = dataset.select(test_inds)
        
        return train_dataset, valid_dataset, test_dataset

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