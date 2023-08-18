import torch
import numpy as np
from torch.autograd import Variable
from harvard_transformer import *


def data_gen(V, batch, nbatches):
    '''
    Generate random data for a src-tgt copy task.

    :param V: vocab size
    :param batch: num of sample in each batch
    :param nbatches: the number of batches to be generated
    :return:
    '''
    for i in range(nbatches):
        # generate data with shape [batch, 10], and value between 1 and vocab size.
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        # data shape: torch.Size([30, 10])
        # 30 means 30 sentences, 10 means each sentence has 10 words.

        data[:, 0] = 1  # set the first column to 1 , meaning the start token
        src = Variable(data, requires_grad=False) # src shape is torch.Size([30, 10])
        tgt = Variable(data, requires_grad=False) # tgt shape is torch.Size([30, 10])
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
V = 11  # vocab size
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)  # the cri
model = make_model(V, V, N=2,d_model=1024)  # create transformer
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 30, 20), model,
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model,
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
        print('# prob shape is ', prob.shape)
        print('# prob is ', prob)
        _, next_word = torch.max(prob, dim=1)# return value and index
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
src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))  # shape is [1, 10] : one sentence, 10 words
src_mask = Variable(torch.ones(1, 1, 10))  # shape is [1, 1, 10] ,[batch, 1, seq_len], meaning no padding
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
