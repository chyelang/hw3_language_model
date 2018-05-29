import os
import torch

class Corpus(object):
    def __init__(self, path, batch_size, mode='train'):
        self.vocabulary = []
        self.word_id = {}
        print('preparing corpus...')
        self.batch_size = batch_size
        if mode == 'train':
            self.train = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
            self.train_batch_num = self.train.size(0) // self.batch_size["train"]
            self.valid_batch_num = self.valid.size(0) // self.batch_size["valid"]
            self.train = self.train.narrow(0, 0, self.batch_size["train"] * self.train_batch_num)
            self.valid = self.valid.narrow(0, 0, self.batch_size["valid"] * self.valid_batch_num)
            self.train = self.train.view(self.batch_size["train"], -1).t().contiguous()
            self.valid = self.valid.view(self.batch_size["valid"], -1).t().contiguous()
        elif mode == "test":
            self.test = self.tokenize(path)
            self.test_batch_num = self.test.size(0) // self.batch_size["test"]
            self.test = self.test.narrow(0, 0, self.batch_size["test"] * self.test_batch_num)
            self.test = self.test.view(self.batch_size["test"], -1).t().contiguous()
        else:
            print("error: invalid mode")

    def tokenize(self, file_name):
        assert os.path.exists(file_name)
        file_lines = open(file_name, 'r').readlines()
        num_of_words = 0
        for line in file_lines:
            words = line.split() + ['<eos>']
            num_of_words += len(words)
            for word in words:
                if word not in self.word_id:
                    self.word_id[word] = len(self.vocabulary)
                    self.vocabulary.append(word)
        file_tokens = torch.LongTensor(num_of_words)
        token_id = 0
        for line in file_lines:
            words = line.split() + ['<eos>']
            for word in words:
                file_tokens[token_id] = self.word_id[word]
                token_id += 1
        return file_tokens
