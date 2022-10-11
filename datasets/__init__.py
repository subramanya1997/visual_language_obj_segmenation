from datasets.a2d_sentences.a2d_sentences_dataset import A2DSentencesDataset
from datasets.a2d_sentences.a2d_sentences_feature_extractor import A2DSentencesDatasetFeature

def build_dataset(subset_type, dataset_name, running_mode='train',**kwargs):
    if dataset_name == 'a2d_sentences':
        if running_mode == 'feature_extractor':
            return A2DSentencesDatasetFeature(subset_type=subset_type, **kwargs)
        return A2DSentencesDataset(subset_type=subset_type, **kwargs)
    else:
        assert False, f'Error: dataset {dataset_name} is not supported'