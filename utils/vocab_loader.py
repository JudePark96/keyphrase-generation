__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import codecs
from typing import Dict

from tqdm import tqdm

from utils.constants import (
    BOS_WORD,
    EOS_WORD,
    DIGIT_WORD,
    PAD_WORD,
    SEP_WORD,
    UNK_WORD
)


def load_vocab(src_filename, vocab_size=None) -> Dict[str, int]:
    vocab2idx = {}

    with codecs.open(src_filename, 'r', 'utf-8') as f:
        for idx, word in tqdm(enumerate(f)):
            if word not in vocab2idx:
                vocab2idx[word.strip()] = idx

            if vocab_size and len(vocab2idx) >= vocab_size:
                break
        f.close()

    if PAD_WORD not in vocab2idx:
        raise ValueError('<pad> is not in vocab')
    if UNK_WORD not in vocab2idx:
        raise ValueError('<unk> char is not in vocab')
    if BOS_WORD not in vocab2idx:
        raise ValueError('<s> char is not in vocab')
    if EOS_WORD not in vocab2idx:
        raise ValueError('</s> char is not in vocab')
    if DIGIT_WORD not in vocab2idx:
        raise ValueError('digit is not in vocab')
    if SEP_WORD not in vocab2idx:
        raise ValueError('<sep> is not in vocab')

    return vocab2idx


if __name__ == '__main__':
    print(load_vocab('./vocab.txt'))