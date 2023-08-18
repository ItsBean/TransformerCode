import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

seaborn.set_context(context="talk")


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.

    encoder: the encoder
    decoder: the decoder
    src_embed: embedding layer for source, function is to convert the encoder input to embedding.
    tgt_embed: embedding layer for target, function is to convert the decoder input to embedding.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        '''


        :param memory: the output of encoder
        :param src_mask:  the mask of padding token.
        :param tgt: target input (the last token of the target is deleted.)
        :param tgt_mask: mask padding and future token.
        :return:
        '''

        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    '''
    Define standard linear + softmax generation step.
    the generator is just a linear layer with softmax
    '''

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    '''
    Construct a layernorm module (See citation for details).
    layernorm is to make data has zero mean and unit variance
    and add A and B to make the data more flexible
    '''

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# class SublayerConnection(nn.Module):
#     """


#     A residual connection followed by a layer norm.
#     Note for code simplicity the norm is first as opposed to last.
#     """
#     def __init__(self, size, dropout):
#         super(SublayerConnection, self).__init__()
#         self.norm = LayerNorm(size)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, sublayer):
#         "Apply residual connection to any sublayer with the same size."
#         return x + self.dropout(sublayer(self.norm(x)))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Following the original Transformer paper's approach.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        # x shape torch.Size([30, 9, 512])
        # m shape torch.Size([30, 10, 512])

        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))  #
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


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


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # query shape torch.Size([30, 8, 10, 64])
    # key shape torch.Size([30, 8, 10, 64])
    # value shape torch.Size([30, 8, 10, 64])


    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        # query shape torch.Size([30, 10, 512])
        # key shape torch.Size([30, 10, 512])
        # value shape torch.Size([30, 10, 512])


        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    '''
    d_model: the dimension of the input and output
    d_ff: the dimension of the hidden layer


    '''

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    '''
    the embedding layer will also be trained like other layers
    '''

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # 5000, 512
        position = torch.arange(0, max_len).unsqueeze(1)  # 5000, 1
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))  # e
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 1, 5000, 512
        self.register_buffer('pe', pe)  #

    def forward(self, x):
        '''
        x shape is [batch_size, seq_len, d_model]
        The broadcast rule will change the shape of pe to [batch_size, seq_len, d_model]
        :param x:
        :return:
        '''
        x = x + Variable(self.pe[:, :x.size(1)],  # pe is 1, x.length, 512
                         requires_grad=False)
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    '''
    "Helper: Construct a Transformer model from hyperparameters."

    :param src_vocab: vocab size of source language
    :param tgt_vocab: vocab size of target language
    :param N: number of encoder and decoder layers
    :param d_model: representation dimension
    :param d_ff: ffn's hidden layer dimension
    :param h: multi-head attention's head number
    :param dropout:
    :return:
    '''
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)  # position encoding
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),  # src embedding
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),  # tgt embedding
        Generator(d_model, tgt_vocab)
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:  # only reset the weight of linear layer
            nn.init.xavier_uniform(p)
    return model


class Batch:
    '''
    src: shape is [sentence_num, src_sentence_length]
    trg: shape is [sentence_num, tgt_sentence_length]
    src mask: shape is [sentence_num, 1, src_sentence_length]
    trg mask: shape is [sentence_num, tgt_sentence_length, tgt_sentence_length]
    in trg mask, each sentence mask containing padding mask and future word mask.

    Object for holding a batch of data with mask during training.
    the class is designed for the training data
    src_mask is used to mask the padding token
    trg_mask is used to mask the future token and padding token

    '''

    def __init__(self, src, trg=None, pad=0):
        self.src = src  # shape is [batch_size, seq_len],
        self.src_mask = (src != pad).unsqueeze(-2)  # shape is [batch_size, 1, seq_len]
        if trg is not None:
            self.trg = trg[:, :-1]  # delete the last token, this is the input of decoder e.g : [0,9]
            self.trg_y = trg[:, 1:]  # delete the first token, this is the output of decoder e.g : [1,10]
            # for example :

            self.trg_mask = \
                self.make_std_mask(self.trg, pad)  # shape is [batch_size, seq_len, seq_len]
            # trg_mask[0][1][1]=true means the first batch, the second token can attend to the second token

            self.ntokens = (self.trg_y != pad).data.sum()  # the number of tokens in the batch.

    @staticmethod
    def make_std_mask(tgt, pad):
        '''
        Create a mask to hide padding and future words.
        tgt shape is [batch_size, seq_len]
        example input:
        a = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

        example output:
        tensor([[[ True, False, False, False],
         [ True,  True, False, False],
         [ True,  True,  True, False],
         [ True,  True,  True, False]],

        [[ True, False, False, False],
         [ True,  True, False, False],
         [ True,  True, False, False],
         [ True,  True, False, False]]])





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


def run_epoch(data_iter, model, loss_compute):
    '''
    data_iter : the data iterator
    model : the model
    loss_compute : the loss function


    batch.trg_y : [batch_size, trg_seq_len - 1]
    out : [batch_size, trg_seq_len - 1, vocab_size]
    out[0][1][1] means the first batch, the second token, the second token(in the vocab)'s probability
    Standard Training and Logging Function

    '''
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)

        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    '''

    "Keep augmenting batch and calculate total number of tokens + padding."
    The batch_size_fn function seems to compute a dynamic batch size based on
    the source and target sequence lengths within the batch.

    Dynamic batching allows you to adjust batch size based on the actual sequence lengths in the batch,
    ensuring that each batch roughly uses the same amount of computational resources.
    :param new:
    :param count:
    :param sofar:
    :return:

    '''

    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))  # max sentence length in this batch
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)  # max sentence length in this batch
    src_elements = count * max_src_in_batch  # total number of tokens in this batch
    tgt_elements = count * max_tgt_in_batch  # total number of tokens in this batch
    return max(src_elements, tgt_elements)


class NoamOpt:
    '''
    "Optim wrapper that implements rate." \\

    the NoamOpt sets a limit step, before reaching the limit step, the learning rate increases linearly.
    After reaching the limit step, the learning rate decreases proportionally to the inverse square root of the step number.

    model_size: the d_model in the paper. the embedding size and the hidden size of the encoder and decoder.
    factor: the coefficient of the learning rate.
    warmup: the number of steps to increase the learning rate linearly.
    optimizer: the optimizer, such as Adam.

    '''

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup # hyperparameter
        self.factor = factor # hyperparameter
        self.model_size = model_size
        self._rate = 0

    def step(self):
        '''
        Update parameters and rate
        '''
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        '''
        Implement `lrate` above
        the learning rate goes up linearly.
        and then goes down
        '''
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    '''
    this is an example
    :param model:
    :return:
    '''
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    '''
    "Implement label smoothing."

    this loss will ignore the padding token's prediction.

    '''

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx # the index of pad token
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size # vocab size
        self.true_dist = None

    def forward(self, x, target):
        '''
        example:
        x = [[-1.2, 2.5, -0.5, 1.1, -0.3],   # logits for first token
        [-1.5, -0.5, 2.3, 0.1, -1.1]]   # logits for second token
        x shape is [2, 5]
        target = [1, 2]  # The true labels for these sequences are 'a' and 'b', respectively.
        target shape is [batch_size, 1]

        for pad token, the true dist will be [0, 0, 0, 0, 0]
        the reason:

        returned value is criterion
        :param x:
        :param target:
        :return:
        '''

        assert x.size(1) == self.size
        true_dist = x.data.clone() # initial the true dist
        true_dist.fill_(self.smoothing / (self.size - 2)) # fill each element with smooth value.
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        '''
        true dist will be like this:
        [[0.0333, 0.9, 0.0333, 0.0333, 0.0333],
        [0.0333, 0.0333, 0.9, 0.0333, 0.0333]]
        '''

        true_dist[:, self.padding_idx] = 0 # set the pad token's probability to 0
        mask = torch.nonzero(target.data == self.padding_idx) # find the pad token's index
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        # kl = sum(p * log(p/q))  , and x is q, true_dist is p
        # by setting pad target as all 0, we can ignore the pad token's loss
        # because kl(target = pad) = 0,
        return self.criterion(x, Variable(true_dist, requires_grad=False))
