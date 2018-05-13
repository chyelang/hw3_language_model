# coding: utf-8
import argparse
import time
from datetime import datetime
import math
import torch
import torch.nn as nn
import torch.optim as optim

import data
import model
import os

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--data', type=str, default='./data/ptb',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='GRU',
                    help='type of recurrent net (LSTM, GRU)')
parser.add_argument('--ninp', type=int, default=200,
                    help='size of word embeddings(input)')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=3,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--max_sql', type=int, default=35,
                    help='sequence length for bptt')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', type=bool, default=True,
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', type=bool, default=False,
                    help='use CUDA')
parser.add_argument('--save_file', type=str, default='./saved_model/model.pt',
                    help='path to save the final model')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')

args = parser.parse_args()
if not os.path.exists(os.path.dirname(args.save_file)):
    os.mkdir(os.path.dirname(args.save_file))

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
if args.cuda:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

train_batch_size = args.batch_size
# valid_batch_size = args.batch_size
valid_batch_size = args.batch_size*2

# load train and valid data
corpus = data.Corpus(args.data, {"train": train_batch_size, "valid": valid_batch_size})
log_interval = len(corpus.train) // args.max_sql // 15

# build the model
nvoc = len(corpus.word_id)
model = model.LMModel(args.model, nvoc, args.ninp, args.nhid, args.nlayers, args.batch_size, args.dropout, args.tied).to(device)
criterion = nn.CrossEntropyLoss()
lr = args.lr
opt = optim.Adam(model.parameters(), lr=lr)

def get_batch(data_source, i):
    start_index = i
    seq_len = min(args.max_sql, data_source.size(0) - i - 1)
    data_loader = data_source

    data = data_loader[start_index:start_index + seq_len, :]
    target = data_loader[start_index + 1:start_index + seq_len + 1, :].view(-1)
    return data.to(device), target.to(device)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.
def evaluate(corpus):
    data_source = corpus.valid
    model.eval()
    total_loss = 0.
    nvoc = len(corpus.word_id)
    # hidden = model.init_hidden(valid_batch_size)
    # hidden = get_hidden0(model.hidden0)
    # hidden = model.hidden0.repeat(1, valid_batch_size, 1)
    hidden = None
    with torch.no_grad():
        for i in range(0, corpus.valid.size(0) - 1, args.max_sql):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, nvoc)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)

# Train Function
def train(corpus, opt):
    # Turn on training mode which enables dropout.
    data_source = corpus.train
    model.train()
    total_loss = 0.
    start_time = time.time()
    nvoc = len(corpus.word_id)

    # print(next(model.named_parameters()))
    #从这里来看init hidden weight在每个epoch都被重新初始化了？
    # hidden = model.init_hidden(train_batch_size)
    # hidden = get_hidden0(model.hidden0)
    # hidden = model.hidden0.repeat(1, train_batch_size, 1)
    hidden = None
    # 应该是因为文本数量够多，所以range不长等于bptt，使得每个batch之间没有重合？
    for batch, i in enumerate(range(0, data_source.size(0) - 1, args.max_sql)):
        data, targets = get_batch(corpus.train, i)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, nvoc), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        opt.step()
        # print(next(model.named_parameters()))
        total_loss += loss.item()
        hidden = repackage_hidden(hidden)

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            try:
                ppl = math.exp(cur_loss)
            except OverflowError:
                ppl = float('inf')
            print('{} | epoch {:3d} | {:5d}/{:5d} batches | lr {:06.5f} | {:5.2f} ms/batch | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(datetime.now().strftime('%m-%d %H:%M:%S'),
                epoch, batch, len(data_source) // args.max_sql, lr,
                elapsed * 1000 / log_interval, cur_loss, ppl))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
best_val_loss = None
patience = 5
wait = 0
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)
for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train(corpus, opt)
    val_loss = evaluate(corpus)
    try:
        ppl = math.exp(val_loss)
    except OverflowError:
        ppl = float('inf')
    print('-' * 106)
    print('{} | end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(datetime.now().strftime('%m-%d %H:%M:%S'), epoch, (time.time() - epoch_start_time),
                                     val_loss, ppl))
    print('-' * 106)
    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        with open(args.save_file, 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("early stop!")
            break
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        if lr > 1e-5:
            lr /= 2.0
            print("lr = {}".format(lr))
            opt = optim.Adam(model.parameters(), lr=lr)
    # use variant batch_size
    if train_batch_size < 200:
        train_batch_size = int(train_batch_size*2)
        if train_batch_size > 200:
            train_batch_size = 200
        valid_batch_size = train_batch_size*2
        corpus = data.Corpus(args.data, {"train": train_batch_size, "valid": valid_batch_size})
        log_interval = len(corpus.train) // args.max_sql // 15
        print("train_batch_size = {0}, log_interval = {1}".format(train_batch_size, log_interval))

# Load the best saved model.
with open(args.save_file, 'rb') as f:
    model = torch.load(f)
    model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(corpus)
print('=' * 106)
print('{} | End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    datetime.now().strftime('%m-%d %H:%M:%S'), test_loss, math.exp(test_loss)))
print('=' * 106)

