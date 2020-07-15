__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import argparse
import codecs
import json
import re
import string
from collections import Counter

from itertools import chain
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tqdm import tqdm

from utils.constants import (
    BOS_WORD,
    EOS_WORD,
    DIGIT_WORD,
    PAD_WORD,
    SEP_WORD,
    UNK_WORD
)


class DataPreprocessor(object):
    num_and_punc_regex = re.compile(r'[_\-—<>{,(?\\.\'%]|\d+([.]\d+)?', re.IGNORECASE)
    num_regex = re.compile(r'\d+([.]\d+)?')

    def __init__(self, args) -> None:
        super(DataPreprocessor, self).__init__()
        self.src_filename = args.src_filename
        self.dest_filename = args.dest_filename
        self.dest_vocab_path = args.dest_vocab_path
        self.vocab_size = args.vocab_size
        self.parallel_count = args.parallel_count
        self.is_src_lower = args.src_lower
        self.is_src_stem = args.src_stem
        self.is_target_lower = args.target_lower
        self.is_target_stem = args.target_stem
        self.stemmer = PorterStemmer()

    def build_vocab(self, tokens) -> list:
        """Building Vocabulary for Deep Learning"""
        vocab = [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD, DIGIT_WORD, SEP_WORD]
        vocab.extend(list(string.digits))

        token_counter = Counter(tokens).most_common(self.vocab_size)
        for token, count in token_counter:
            # word in vocab can not be duplicated.
            if token not in vocab:
                vocab.append(token)

            if len(vocab) >= self.vocab_size:
                break
        return vocab


    def tokenize(self, text, is_lower, is_stem) -> list:
        """Tokenize the source text."""
        text = self.num_and_punc_regex.sub(r' \g<0> ', text)
        # remove line breakers
        text = re.sub(r'[\r\n\t]', ' ', text)

        # pad spaces to the left and right of special punctuations
        text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', text)

        # tokenize by non-letters (new-added + # & *, but don't pad spaces, to make them as one whole word)
        tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,#&\+\*\(\)\.\'%]', text))

        # replace digit to special token.
        tokens = [w if not re.match('^\d+$', w) else DIGIT_WORD for w in tokens]

        if is_lower:
            tokens = [token.lower() for token in tokens]
        if is_stem:
            tokens = [self.stemmer.stem(token) for token in tokens]

        return tokens

    def process(self):
        keywords, abstracts, titles = [], [], []

        # Step 1 => Read file and tokenize.
        with codecs.open(self.src_filename, 'r', 'utf-8') as f:
            for idx, line in tqdm(enumerate(f)):
                json_object = json.loads(line)
                abstract = json_object['abstract']
                keyword = json_object['keyword']
                title = json_object['title']

                abstract_tokens = self.tokenize(abstract, self.is_src_lower, self.is_src_stem)
                title_tokens = self.tokenize(title, self.is_src_lower, self.is_src_stem)

                keywords.append([self.tokenize(k, self.is_target_lower, self.is_target_stem) for k in keyword.split(';')])
                abstracts.append(abstract_tokens)
                titles.append(title_tokens)

            f.close()

        # Step 2 => Saving resources.
        if self.dest_vocab_path:
            # Vocab => keyword + abstract + title. Only at Train Dataset.
            flatten_keyword_tokens = list(chain(*list(chain(*keywords))))
            abstract_title_tokens = list(chain(*abstracts)) + list(chain(*titles))
            all_vocab_candidates = flatten_keyword_tokens + abstract_title_tokens
            vocab = self.build_vocab(all_vocab_candidates)

            with codecs.open(self.dest_vocab_path, 'w', 'utf-8') as f:
                f.writelines([word + '\n' for word in vocab])
                f.close()

        if self.dest_filename:
            features = []

            for abstract, keyword, title in tqdm(zip(abstracts, keywords, titles)):
                json_object = {
                    'abstract_tokens': abstract,
                    'keyword_tokens': keyword,
                    'title_tokens': title
                }

                features.append(json_object)

            with codecs.open(self.dest_filename, 'w', 'utf-8') as f:
                f.writelines([str(json_object) + '\n' for json_object in features])
                f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_filename', type=str, required=True,
                        help='input source kp20k file path')
    parser.add_argument('-dest_filename', type=str, required=True,
                        help='destination of processed file path')
    parser.add_argument('-dest_vocab_path', type=str,
                        help='')
    parser.add_argument('-vocab_size', type=int, default=50000,
                        help='')
    parser.add_argument('-parallel_count', type=int, default=10)
    parser.add_argument('-src_lower', action='store_true')
    parser.add_argument('-src_stem', action='store_true')
    parser.add_argument('-target_lower', action='store_true')
    parser.add_argument('-target_stem', action='store_true')

    args = parser.parse_args()
    print(args.target_stem)
    processor = DataPreprocessor(args)
    processor.process()


if __name__ == '__main__':
    main()
