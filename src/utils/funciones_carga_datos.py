from datasets import load_dataset


def load_filter_dataset_HuggingFace(dataset_name, n_subset = None, split = None):
    """
    Para datasets HuggingFace
    """
    
    dataset = load_dataset(dataset_name, split=split)
    if n_subset is not None:
        dataset = dataset.select(range(n_subset))
    
    return dataset