# [[1, 1, 1, 0],
#  [1, 1, 0, 0]]
import torch
import numpy as np
from torch.autograd import Variable

a = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])


# a = a.unsqueeze(-2)


# a.unsqueeze(-2)

def subsequent_mask(size):
    '''Mask out subsequent positions.
    this function create lower triangular matrix
    example of subsequent_mask(3)
    :
    tensor([[[1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]]], dtype=torch.uint8)



    '''

    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(tgt, pad):
    '''
    Create a mask to hide padding and future words.
    tgt shape is [batch_size, seq_len]



    '''
    tgt_mask = (tgt != pad).unsqueeze(-2)  # shape is [batch_size, 1, seq_len], pad position is false
    tgt_mask = tgt_mask & Variable(
        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    # [batch_size, 1, seq_len] & [1, seq_len, seq_len], the second param is lower triangular matrix
    '''
    example:
    tgt_mask = 
    [
    [[1,1,1,0]] 
    [[1,1,0,0]]

    lower triangular matrix = 1x4x4


    tgt_mask & lower triangular matrix  shape is [batch_size, seq_len, seq_len]

    the new tgt_mask is expressed as:
    tgt_mask[0] means the first sentence
    and each sentence is expressed as a seq_len x seq_len matrix.
    this matrix is used to mask the attention weight matrix.


    example mask matrix:
    [1000]
    [1100]
    [1110]
    [1110]
    this means that the forth token can attend to the first three tokens.
    because the sentence is [1110], the last is pad, so the forth token can not attend to the last token.

    '''
    return tgt_mask


print(make_std_mask(a, 0))
