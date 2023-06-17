from math import ceil

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp

import torchvision
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, pad_sequence


class DCNN(nn.Module):
    def __init__(self, embedding_shape: tuple, num_features=300):
        super().__init__()

        self.num_features = num_features

        self.embedding

        self.top_k = 4
        self.dmp = DynamicMaxPool(top_k=self.top_k)
        self.ks1 = 7
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=3*num_features, kernel_size=self.ks1,
                               padding=self.ks1-1, groups=num_features//2)
        self.ks2 = (6, 5)
        self.conv2 = nn.Conv2d(in_channels=num_features//2, out_channels=14*num_features//4,
                               kernel_size=self.ks2, padding=(0, self.ks2[1]-1), groups=num_features//4)
        self.fc = nn.Linear(in_features=self.top_k*14*num_features//4, out_features=1)

    def forward(self, x):
        # batch_size x seq_len x dim
        x, lengths = pad_packed_sequence(x, batch_first=True)
        batch_size = x.shape[0]
        dim = x.shape[2]

        # batch_size x dim x seq_len
        x = x.transpose(1, 2)
        # batch_size x out_channels1 x seq_len+6
        x = self.conv1(x)
        # batch_size x seq_len+6 x out_channels1
        x = x.transpose(1, 2)
        # batch_size x max_pool_dim x out_channels1
        x, pool_result_ranges = self.dmp(x, lengths, 1, 2, pool_ranges=lengths+(self.ks1-1))
        # batch_size x out_channels1 x max_pool_dim
        x = x.transpose(2, 1)
        # batch_size x dim//2 x num_maps x max_pool_dim
        x = x.view(batch_size, dim//2, x.shape[1]//(dim//2), x.shape[-1])
        x = torch.tanh(x)
        # batch_size x ? x 1 x out_channels2
        x = self.conv2(x)
        # batch_size x dim//4 x num_maps x seq_len
        x = x.view(batch_size, dim//4, x.shape[1]//(dim//4), x.shape[-1])
        # batch_size x seq_len x num_maps x dim//4
        x = x.transpose(1, 3)
        # batch_size x max_pool_dim x num_maps x dim//4, aligned
        x, _ = self.dmp(x, lengths, 2, 2, pool_ranges=numpy.array(pool_result_ranges)+(self.ks2[1]-1))
        x = torch.tanh(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        x = x.view(-1, 1)
        neg_vec = torch.zeros_like(x)
        return torch.cat((neg_vec, x), 1)


def kmax_pooling(x, k, dim=0):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


class DynamicMaxPool(nn.Module):
    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k

    def forward(self, x, lengths, layer, total_layers, pool_ranges):
        results = []
        for sample, length, pool_range in zip(x, lengths, pool_ranges):
            # pool_range x out_channels
            sample = sample[: pool_range]
            pool_target = max(self.top_k, ceil((total_layers - layer) / total_layers * length))
            sample = kmax_pooling(sample, pool_target, dim=0)
            results.append(sample)
        pool_result_ranges = [len(result) for result in results]
        results = pad_sequence(results, batch_first=True)
        return results, pool_result_ranges


def _test():
    from sentiment140 import Sentiment140Dataset
    dataset = Sentiment140Dataset(data_fp='training.csv')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, collate_fn=Sentiment140Dataset.collate_fn)
    input, label = iter(dataloader).__next__()
    print(dataset.num_words)



if __name__ == '__main__':
    _test()
