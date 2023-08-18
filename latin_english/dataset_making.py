from datasets import load_dataset
from datasets import load_from_disk
from transformers import BertTokenizer
import torch
import numpy as np
from torch.autograd import Variable
from harvard_transformer import *

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
# load
tokenized_train_dataset = load_from_disk('./data/latin_english_train_tokenized')
tokenized_test_dataset = load_from_disk('./data/latin_english_test_tokenized')

# tokenized_train_dataset:['id', 'la', 'en', 'file', 'latin_input_ids', 'latin_attention_mask', 'english_input_ids', 'english_attention_mask']

latin_idxs = tokenized_train_dataset['latin_input_ids']
# to torch
latin_idxs_torch = torch.tensor(np.array(latin_idxs))

english_idxs = tokenized_train_dataset['english_input_ids']
# to torch
english_idxs_torch = torch.tensor(np.array(english_idxs))

# test set
latin_idxs_test = tokenized_test_dataset['latin_input_ids']
# to torch
latin_idxs_torch_test = torch.tensor(np.array(latin_idxs_test))


# latin src mask:
latin_src_mask_test = tokenized_test_dataset['latin_attention_mask']
# to torch
latin_src_mask_torch_test = torch.tensor(np.array(latin_src_mask_test))
# [batch,sentence_length] -> [batch,1,sentence_length]
latin_src_mask_torch_test = latin_src_mask_torch_test.unsqueeze(1)
# latin_src_mask_torch_test torch.Size([1014, 1, 128])


# eng
english_idxs_test = tokenized_test_dataset['english_input_ids']
# to torch
english_idxs_torch_test = torch.tensor(np.array(english_idxs_test))

sentence_length = len(latin_idxs_torch[0])
current_batch_index = 0
batch_epoch = 128
nbatches = 50#len(latin_idxs_torch) // batch_epoch
#print param:
print('sentence_length:', sentence_length)
print('current_batch_index:', current_batch_index)
print('batch_epoch:', batch_epoch)
print('nbatches:', nbatches)



def data_gen(V, batch, nbatches):
    '''
    Generate random data for a src-tgt copy task.

    :param V: vocab size
    :param batch: num of sample in each batch
    :param nbatches: the number of batches to be generated
    :return:
    '''
    for i in range(nbatches):
        current_batch_src = latin_idxs_torch[current_batch_index:current_batch_index + batch]
        current_batch_tgt = english_idxs_torch[current_batch_index:current_batch_index + batch]
        src = Variable(current_batch_src, requires_grad=False)  # src shape is torch.Size([30, 10])
        tgt = Variable(current_batch_tgt, requires_grad=False)  # tgt shape is torch.Size([30, 10])
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    '''
    "A simple loss compute and train function."

    the generator : the last layer of the transformer, which is a linear layer.
    criterion : the loss function

    '''

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)  # x shape is [sentence, token num,probability of each token]
        # x shape is torch.Size([30, 9, 11])

        # x.contiguous().view(-1, x.size(-1)) shape is torch.Size([270, 11])

        # y.contiguous().view(-1) shape is  torch.Size([270])
        # y shape torch.Size([30, 9])

        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # return loss.data[0] * norm
        return loss.item() * norm


# Train the simple copy task.
V = tokenizer.vocab_size  # vocab size
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)  # the cri
model = make_model(V, V, N=2)  # create transformer
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, batch_epoch, nbatches), model,
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, batch_epoch, nbatches), model,
                    SimpleLossCompute(model.generator, criterion, None)))


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    '''
    it means that select the most possible token as the next token.
    :param model:
    :param src:
    :param src_mask:
    :param max_len:
    :param start_symbol:
    :return:
    '''
    memory = model.encode(src, src_mask)  # memory shape is [1, 10, 512]
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)  # ys(src_mask) shape is [1, 1]
    # ys shape is  torch.Size([1, 1])
    # subsequent_mask(ys.size(1)) shape is  torch.Size([1, 1, 1])

    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),  # ys shape is [1, 1] , one sentence, one word.
                           Variable(subsequent_mask(ys.size(1))  # tgt_mask shape is [1, 1, 1] # batch, seq_len, seq_len
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        # prob shape is  torch.Size([1, 11]) 11 is vocab size.
        # prob is  tensor([[-13.1888,  -5.1260,  -0.9495,  -0.5367,  -5.2774,  -7.4362,  -6.3639,
        # -7.4378, -5.3054, -4.7504, -7.1342]],
        # print('# prob shape is ', prob.shape)
        # print('# prob is ', prob)
        _, next_word = torch.max(prob, dim=1)  # return value and index
        # print('# torch.max(prob, dim = 1) is ', torch.max(prob, dim=1))
        # torch.max(prob, dim = 1) is  torch.return_types.max(
        # values = tensor([-0.5367], grad_fn= < MaxBackward0 >),
        # indices = tensor([3]))

        next_word = next_word.data[0]
        # print('# ys before cat is ', ys)
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        # print('# ys after cat is ', ys)
        # ys before cat is  tensor([[1]])
        # ys after cat is  tensor([[1, 3]])

        # grad_fn = < LogSoftmaxBackward0 >)

    return ys


model.eval()
# src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))  # shape is [1, 10] : one sentence, 10 words
# src_mask = Variable(torch.ones(1, 1, 10))  # shape is [1, 1, 10] ,[batch, 1, seq_len], meaning no padding
print(greedy_decode(model, latin_idxs_torch_test, latin_src_mask_torch_test, max_len=sentence_length, start_symbol=101))
