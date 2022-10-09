from datasets.a2d_sentences.a2d_sentences_dataset import A2DSentencesDataset

def build_dataset(subset_type, dataset_name, **kwargs):
    if dataset_name == 'a2d_sentences':
        return A2DSentencesDataset(subset_type=subset_type, **kwargs)
    else:
        assert False, f'Error: dataset {dataset_name} is not supported'