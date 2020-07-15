__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

from typing import Dict

import torch


from torch.utils.data import Dataset
from utils.constants import UNK_WORD, PAD_WORD


"""
The dataset consists of JSON format. Each variable below is the key of JSON Object. 
"""
SRC_TOKENS = 'abstract_tokens'
TRG_TOKENS = 'keyword_tokens'
TITLE_TOKENS = 'title_tokens'


class KeyphraseDataset(Dataset):
    def __init__(self,
                 features: list,
                 vocab: dict,
                 max_src_seq_len: int,
                 max_trg_seq_len: int,
                 max_title_seq_len: int) -> None:
        super(KeyphraseDataset, self).__init__()
        self.features = features
        self.vocab = vocab
        self.max_src_seq_len = max_src_seq_len
        self.max_title_seq_len = max_title_seq_len
        self.max_trg_seq_len = max_trg_seq_len

    def __getitem__(self, index: int) -> Dict:
        # initialize as <pad>. index of <pad> is always 0 (zero).
        src_tensor = torch.zeros(1, self.max_src_seq_len).long()
        trg_tensor = torch.zeros(1, self.max_trg_seq_len).long()
        title_tensor = torch.zeros(1, self.max_title_seq_len).long()

        # load the sequences.
        src_tokens = self.features[index][SRC_TOKENS]
        trg_tokens = self.features[index][TRG_TOKENS]
        title_tokens = self.features[index][TITLE_TOKENS]

        # should ignore token after maximum sequence length.
        for idx, token in enumerate(src_tokens[:self.max_src_seq_len]):
            if token not in self.vocab:
                src_tensor[0][idx] = self.vocab[UNK_WORD]
            else:
                src_tensor[0][idx] = self.vocab[token]

        # TODO => need some consideration for OOV count, because of copy mecahnism.

        return {
            'abstract_tokens': src_tensor,
            'len_abstract_tokens': len(src_tokens),
        }

        pass

    def __len__(self) -> int:
        pass
