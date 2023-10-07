import os
import json
import copy
import torch

import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from sybil.dataset.utils import process_caption
from sybil.dataset.catalog import DatasetCatalog
from sybil.dataset.concat_dataset import MyConcatDataset
from sybil.dataset.samplers import DistributedBatchSampler, DistributedMultiDatasetBatchSampler


class BaseDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, mm_root_path: str, embed_path: str, dataset_type: str):
        super(BaseDataset, self).__init__()
        self.embed_path = embed_path
        self.mm_path_list, self.caption_list = [], []
        self.dataset_type_list = []

    def __len__(self):  # number of instances
        return len(self.mm_path_list)

    def __getitem__(self, i):
        with open(os.path.join(self.embed_path, str(os.path.basename(self.mm_path_list[i])) + '.npy'), 'rb') as f:
            caption_embs = torch.from_numpy(np.load(f, allow_pickle=True))  # (num_clip_tokens, 768)

        return dict(mm_paths=self.mm_path_list[i], output_texts=self.caption_list[i], caption_embs=caption_embs,
                    dataset_types=self.dataset_type_list[i])

    def collate(self, instances):
        mm_paths, output_texts, caption_embs, dataset_types = tuple(
            [instance[key] for instance in instances] for key in
            ("mm_paths", "output_texts", "caption_embs", "dataset_types"))
        return dict(
            mm_paths=mm_paths,
            output_texts=output_texts,
            caption_embs=caption_embs,
            dataset_types=dataset_types
        )

def load_dataset(args, dataset_name_list):
    """
    Args:
        args:
        dataset_name_list: List[str]
        repeats: List[int], the training epochs for each dataset

    """
    # concat_data = get_concat_dataset(dataset_name_list)
    concat_data = MyConcatDataset(dataset_name_list)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    sampler = torch.utils.data.RandomSampler(concat_data)
    batch_sampler = DistributedMultiDatasetBatchSampler(dataset=concat_data,
                                                        sampler=sampler,
                                                        batch_size=batch_size,
                                                        drop_last=True,
                                                        rank=rank,
                                                        world_size=world_size)
    iter_ = DataLoader(
        concat_data,
        batch_sampler=batch_sampler,
        num_workers=1,
        collate_fn=concat_data.collate,
        pin_memory=True
    )
    return concat_data, iter_, sampler