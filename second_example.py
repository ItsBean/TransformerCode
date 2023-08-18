from harvard_transformer import *
import torch
import numpy as np
from torch.autograd import Variable
from torchtext.legacy import data
from torchtext import datasets

import spacy

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    '''
    example:
    tokenize_en('I am a student')
    ['I', 'am', 'a', 'student']

    :param text:
    :return:
    '''
    return [tok.text for tok in spacy_en.tokenizer(text)]


BOS_WORD = '<s>' # beginning of sentence
EOS_WORD = '</s>' # end of sentence
BLANK_WORD = "<blank>" # padding word
SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                 eos_token=EOS_WORD, pad_token=BLANK_WORD)

MAX_LEN = 100
train, val, test = datasets.IWSLT.splits(
    exts=('.de', '.en'), fields=(SRC, TGT),
    filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                          len(vars(x)['trg']) <= MAX_LEN)
print('# shape of train:', len(train))
print('# shape of val:', len(val))
assert 0

MIN_FREQ = 2
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
