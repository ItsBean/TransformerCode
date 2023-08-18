import torch

example_target = torch.LongTensor([1, 2])

print(example_target.unsqueeze(1))

print('# shape of example_target:', example_target.shape)
print('# shape of example_target.unsqueeze(1):', example_target.unsqueeze(1).shape)

