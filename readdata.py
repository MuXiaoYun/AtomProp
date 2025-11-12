from fairchem.core.datasets import AseDBDataset
dataset_path = "data/omol25/train_4M"
dataset = AseDBDataset({"src": dataset_path})
atoms = dataset.get_atoms(0)
atomic_positions = atoms.positions
atomic_numbers = atoms.get_atomic_numbers()
info = atoms.info

print("Atomic Positions:\n", atomic_positions)
print("Atomic Numbers:\n", atomic_numbers)
print("Additional Info:\n", info)