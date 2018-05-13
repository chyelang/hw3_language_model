import torch
import torch.nn as nn
import modules

class LMModel(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer. 
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding. 
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, rnn_type, nvoc, ninp, nhid, nlayers, batch_size, dropout=0.5, tie_weights=True):
        super(LMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nvoc, ninp)
        self.rnn_type = rnn_type
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
            hidden0 = torch.zeros(2, nlayers, 1, nhid).uniform_(-0.1, 0.1)
            self.hidden0 = nn.Parameter(hidden0, requires_grad=True)
        else:
            # self.rnn = nn.GRU(ninp, nhid, nlayers, dropout=dropout)
            self.rnn = modules.LayerNormGRU(ninp, nhid)
            hidden0 = (torch.zeros(nlayers, 1, nhid).uniform_(-0.1, 0.1))
            self.hidden0 = nn.Parameter(hidden0, requires_grad=True)
        # self.rnn = modules.LayerNormLSTM(ninp, nhid, bias=True, dropout=dropout,
        #               dropout_method='pytorch', ln_preact=True, learnable=True)
        self.decoder = nn.Linear(nhid, nvoc)
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to ninp')
            self.decoder.weight = self.encoder.weight
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers
        # self.layer_norm = nn.LayerNorm(input.size()[1:])

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input, hidden):
        if hidden is None:
            batch_size = input.size(1)
            if self.rnn_type == "LSTM":
                hidden = self.hidden0.repeat(1,1,batch_size,1)
            else:
                hidden = self.hidden0.repeat(1, batch_size, 1)
        embeddings = self.drop(self.encoder(input))
        output, hidden = self.rnn(embeddings, hidden)
        # output = torch.transpose(output, 0, 1)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        # weight = next(self.parameters())
        # if self.rnn_type == 'LSTM':
        #      #这里所说的hidden state包括ht和ct，lstm两者都有，而gru中只有ct
        #      #这里的init weight也被加入到了可学习的参数中？
        #     return (weight.new_zeros(self.nlayers, bsz, self.nhid),
        #             weight.new_zeros(self.nlayers, bsz, self.nhid))
        # else:
        #     return weight.new_zeros(self.nlayers, bsz, self.nhid)

        if self.rnn_type == 'LSTM':
            self.hidden0 = (torch.nn.Parameter(torch.randn(self.nlayers, bsz, self.nhid), requires_grad=True).type(torch.FloatTensor),
            torch.nn.Parameter(torch.randn(self.nlayers, bsz, self.nhid), requires_grad=True).type(torch.FloatTensor))
        else:
            self.hidden0 = (torch.nn.Parameter(torch.randn(self.nlayers, bsz, self.nhid), requires_grad=True).type(torch.FloatTensor))




