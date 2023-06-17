import collections
import copy
import math
import random

import pickle
import re

import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence

from torch.utils.data import Dataset
import os.path as osp
import numpy as np
import datetime
import string
from bs4 import BeautifulSoup

from matplotlib import pyplot as plt

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from abc import ABC, abstractmethod


class Sentiment140Dataset(Dataset):
    def __init__(self, data_fp):
        """
        """
        super().__init__()

        raw_data = []
        with open(data_fp, 'r', encoding='latin1') as f:
            for line_idx, line in enumerate(f.readlines()):
                tup = line.split('","')
                # remove ""
                tup[0] = tup[0][1:]
                tup[-1] = tup[-1][:-1]
                raw_data.append(tup)
                if line_idx > 0:
                    assert len(raw_data[-1]) == len(raw_data[-2])
        self.raw_data = raw_data

        self.num_words = None  # filled in preprocess
        self.processed_data = self.preprocess(self.raw_data)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, item):
        return self.processed_data[item]

    @staticmethod
    def collate_fn(batch):
        tweets, labels = tuple(zip(*batch))
        seq_lens = [len(tweet) for tweet in tweets]
        # sort according to length
        order = np.argsort(seq_lens)[::-1]
        tweets = np.array(tweets)[order]
        labels = np.array(labels)[order]

        tweets = [torch.from_numpy(tweet) for tweet in tweets]
        tweets = pack_sequence(tweets)
        return tweets, torch.from_numpy(labels)
    
    def preprocess(self, raw_data):
        word_map = dict()
        cur_wid = 0
        for idx, (sentiment, tweet_id, dtime, query, client_id, tweet) in enumerate(raw_data):
            if sentiment == '4':
                sentiment = 1
            elif sentiment == '0':
                sentiment = 0
            else:
                raise RuntimeError("sentiment not 0 / 4")

            words = tweet.split()
            for word in words:
                if word not in word_map:
                    word_map[word] = cur_wid
                    cur_wid += 1
            tweet = np.array([word_map[word] for word in words])

            raw_data[idx] = [tweet, sentiment]
        self.num_words = len(word_map)
        return raw_data


def _test():
    dataset = Sentiment140Dataset(data_fp='training.csv')
    print(dataset[0])
    print(len(dataset))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, collate_fn=Sentiment140Dataset.collate_fn)
    print(iter(dataloader).__next__())


if __name__ == "__main__":
    _test()
