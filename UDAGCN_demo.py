# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from argparse import ArgumentParser
from dual_gnn.cached_gcn_conv import CachedGCNConv
from dual_gnn.dataset.DomainData import DomainData
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='acm')
parser.add_argument("--target", type=str, default='dblp')
parser.add_argument("--name", type=str, default='UDAGCN')
parser.add_argument("--seed", type=int, default=200)
parser.add_argument("--encoder_dim", type=int, default=16)

args = parser.parse_args()
seed = args.seed
encoder_dim = args.encoder_dim

id = "source: {}, target: {}, seed: {}, encoder_dim: {}".format(args.source, args.target, seed, encoder_dim)
print(id)

rate = 0.0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
dataset = DomainData("data/{}".format(args.source), name=args.source)
source_data = dataset[0]
print(source_data)
dataset = DomainData("data/{}".format(args.target), name=args.target)
target_data = dataset[0]
print(target_data)

source_data = source_data.to(device)
target_data = target_data.to(device)


class GNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GNN, self).__init__()

        self.dropout_layers = [nn.Dropout(0.1) for _ in range(2)]

        self.conv_layers = nn.ModuleList([
            CachedGCNConv(dataset.num_features, 128, **kwargs),
            CachedGCNConv(128, encoder_dim, **kwargs)
        ])

    def forward(self, x, edge_index, cache_name):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x


loss_func = nn.CrossEntropyLoss().to(device)

encoder = GNN().to(device)

cls_model = nn.Sequential(
    nn.Linear(encoder_dim, dataset.num_classes),
).to(device)

models = [encoder, cls_model]
params = [p for model in models for p in model.parameters()]
optimizer = torch.optim.Adam(params, lr=3e-3)


def gcn_encode(data, cache_name, mask=None):
    encoded_output = encoder(data.x, data.edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


def encode(data, cache_name, mask=None):
    return gcn_encode(data, cache_name, mask)


def predict(data, cache_name, mask=None):
    encoded_output = encode(data, cache_name, mask)
    logits = cls_model(encoded_output)
    return logits


def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    return accuracy


def test(data, cache_name, mask=None):
    for model in models:
        model.eval()
    logits = predict(data, cache_name, mask)
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    accuracy = evaluate(preds, labels)
    return accuracy


epochs = 200


def train(epoch):
    for model in models:
        model.train()
    optimizer.zero_grad()

    encoded_source = encode(source_data, "source")
    source_logits = cls_model(encoded_source)

    # use source classifier loss:
    cls_loss = loss_func(source_logits, source_data.y)

    for model in models:
        for name, param in model.named_parameters():
            if "weight" in name:
                cls_loss = cls_loss + param.mean() * 3e-3

    loss = cls_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


best_source_acc = 0.0
best_target_acc = 0.0
best_epoch = 0.0
for epoch in range(1, epochs):
    train(epoch)
    source_correct = test(source_data, "source", source_data.test_mask)
    target_correct = test(target_data, "target")
    print("Epoch: {}, source_acc: {}, target_acc: {}".format(epoch, source_correct, target_correct))
    if target_correct > best_target_acc:
        best_target_acc = target_correct
        best_source_acc = source_correct
        best_epoch = epoch
print("=============================================================")
line = "{} - Epoch: {}, best_source_acc: {}, best_target_acc: {}".format(id, best_epoch, best_source_acc, best_target_acc)
print(line)
