import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LabelSmoothing(nn.Module):
    '''
    "Implement label smoothing."
    this is the loss function.

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
        x = [[-1.2, 2.5, -0.5, 1.1, -0.3],   # logits for first sequence
        [-1.5, -0.5, 2.3, 0.1, -1.1]]   # logits for second sequence
        x shape is [2, 5]
        target = [1, 2]  # The true labels for these sequences are 'a' and 'b', respectively.
        target shape is [batch_size, 1]

        returned value is criterion
        :param x:
        :param target:
        :return:
        '''

        assert x.size(1) == self.size
        true_dist = x.data.clone() # shape [2,5]
        # value : true dist:  tensor([[-1.2000,  2.5000, -0.5000,  1.1000, -0.3000],
        #         [-1.5000, -0.5000,  2.3000,  0.1000, -1.1000]])


        true_dist.fill_(self.smoothing / (self.size - 2)) # fill each element with smooth value.
        # true dist after fill_: tensor([[0.1333, 0.1333, 0.1333, 0.1333, 0.1333],
        # [0.1333, 0.1333, 0.1333, 0.1333, 0.1333]])
        # shape of true dist after fill_: torch.Size([2, 5])



        # unsuqeeze : torch.Size([2]) -> torch.Size([2, 1])
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true dist after scatter_: tensor([[0.1333, 0.6000, 0.1333, 0.1333, 0.1333],
        # [0.1333, 0.1333, 0.6000, 0.1333, 0.1333]])
        # shape of true dist after scatter_: torch.Size([2, 5])



        true_dist[:, self.padding_idx] = 0 # set the pad token's probability to 0
        # true dist after set pad token to 0:  tensor([[0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
        # [0.0000, 0.1333, 0.6000, 0.1333, 0.1333]])
        # shape of true dist after set pad token to 0:  torch.Size([2, 5])

        mask = torch.nonzero(target.data == self.padding_idx) # find the pad token's index

        # mask: tensor([[1]])
        # shape of mask: torch.Size([1, 1])
        # mask.dim(): 2
        if mask.dim() > 0:
            # index_fill_(dim, index, value) -> Tensor
            # 0 specifies the first dimension,
            # and mask's value specifies the specific row.
            # 0.0 specifies the value to fill.
            # so this code will set the specific row's all value to zero
            true_dist.index_fill_(0, mask.squeeze(), 0.0)


        self.true_dist = true_dist

        '''
        true dist will be like this:
        [[0.0, 0.9, 0.0333, 0.0333, 0.0333],
        [0.0, 0.0333, 0.9, 0.0333, 0.0333]]
        '''

        return self.criterion(x, Variable(true_dist, requires_grad=False))

label_smoothing = LabelSmoothing(5, 0, 0.4)

example_x = torch.FloatTensor([[-1.2, 2.5, -0.5, 1.1, -0.3], [-1.5, -0.5, 2.3, 0.1, -1.1]])
example_target = torch.LongTensor([1, 2])
example_target2 = torch.LongTensor([1, 0])
label_smoothing(example_x, example_target2)


