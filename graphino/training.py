"""
Author: Salva Rühling Cachay
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from eval_gcn import evaluate_preds

from utilities.data_wrangling import load_cnn_data, to_dataloaders
from utilities.utils import get_euclidean_adj


def train_epoch(dataloader, model, criterion, optims, device, epoch, nth_step=100):
    if not isinstance(optims, list):
        optims = [optims]
    model.train()
    total_loss = 0
    for iter, (X, Y) in enumerate(dataloader, 1):
        # X, Y = X.to(device), Y.to(device)
        for optim in optims:
            optim.zero_grad()
        X = X.reshape((X.shape[0], -1, X.shape[3])).transpose(1, 2)  # shape = (batch_size x #features x #nodes)

        preds = model(X)
        loss = criterion(preds, Y)
        loss.backward()
        for optim in optims:
            optim.step()
        total_loss += loss.item()
    num_edges = torch.count_nonzero(model.adj.detach()).item()
    return total_loss / iter, num_edges


def evaluate(dataloader, model, device, return_preds=False):
    model.eval()
    total_loss_l2 = 0
    total_loss_l1 = 0
    preds = None
    Ytrue = None
    for i, (X, Y) in enumerate(dataloader, 1):
        assert len(X.size()) == 4, "Expected X to have shape (batch_size, #channels, window, #nodes)"
        # X, Y = X.to(device), Y.to(device)
        X = X.reshape((X.shape[0], -1, X.shape[3])).transpose(1, 2)
        with torch.no_grad():
            output = model(X)
        if preds is None:
            preds = output
            Ytrue = Y
        else:
            preds = torch.cat((preds, output))
            Ytrue = torch.cat((Ytrue, Y))
        total_loss_l2 += F.mse_loss(output, Y).item()
        total_loss_l1 += F.l1_loss(output, Y).item()

    preds = preds.data.cpu().numpy()
    Ytest = Ytrue.data.cpu().numpy()
    oni_stats = evaluate_preds(Ytest, preds, return_dict=True)
    oni_stats['mae'] = total_loss_l1
    if return_preds:
        return total_loss_l2 / i, oni_stats, Ytest, preds
    else:
        return total_loss_l2 / i, oni_stats


def get_static_feats(params, net_params, coordinates, trainset):
    max_lat = max(params['lat_max'], params['lat_min'])
    static_feats = np.array([
        [lat / max_lat, (lon - 180) / 360] for lat, lon in coordinates
    ])  # (#nodes, 2) = (#nodes (lat, lon))
    return static_feats


def get_dataloaders(params, net_params):
    # Load data
    load_data_kwargs = {
        'window': params['window'], 'lead_months': params['horizon'], 'lon_min': params['lon_min'],
        'lon_max': params['lon_max'], 'lat_min': params['lat_min'], 'lat_max': params['lat_max'],
        'data_dir': params['data_dir'], 'use_heat_content': params['use_heat_content'],
        'add_index_node': net_params['index_node']
    }
    dataset = np.load("Data/Seasfire/dataset.npy")
    y = np.load("Data/Seasfire/y.npy")
    cords = np.load("Data/Seasfire/cords.npy")
    net_params['num_nodes'] = dataset.shape[3]
    adj = None
    static_feats = get_static_feats(params, net_params, cords, dataset)

    trainloader, valloader, testloader = \
        to_dataloaders(dataset,y, batch_size=params['batch_size'],
                       valid_split=params['validation_frac'], concat_cmip5_and_soda=True,
                       shuffle_training=params['shuffle'], validation=params['validation_set'])


    return (adj, static_feats, cords), (trainloader, valloader, testloader)


def get_dirs(params, net_params):
    suffix = params['ID'] + f"{net_params['num_nodes']}nodes_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    params['ID'] = suffix
    log_dir = f"{params['model_dir']}/logs/" + suffix
    ckpt_dir = f"{params['model_dir']}/checkpoints/" + suffix
    for d in [log_dir, ckpt_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    return ckpt_dir, log_dir
