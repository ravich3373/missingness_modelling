#!/usr/bin/env python
# Written by Minh Nguyen and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
from __future__ import print_function, division
import argparse
import json
import time
import pickle

import numpy as np
import torch

import misc as misc
from model import MODEL_DICT

from transformer import Transformer
from typing import Tuple, Optional
import torch.distributions as tdist

from training_objectives import negative_log_likelihood


def kl_divergence(
    z1: Tuple[torch.Tensor, torch.Tensor],
    z2: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    mean1, logvar1 = z1

    if z2 is not None:
        mean2, logvar2 = z2
    else:
        mean2 = torch.zeros_like(mean1)
        logvar2 = torch.zeros_like(logvar1)

    sigma1 = logvar1.exp().sqrt()
    sigma2 = logvar2.exp().sqrt()

    normal1 = tdist.Normal(mean1, sigma1)
    normal2 = tdist.Normal(mean2, sigma2)

    kld = tdist.kl_divergence(normal1, normal2)
    kld = kld.sum(axis=1)
    return kld


def ent_loss(pred, true, mask):
    """
    Calculate cross-entropy loss
    Args:
        pred: predicted probability distribution,
              [nb_timpoints, nb_subjects, nb_classes]
        true: true class, [nb_timpoints, nb_subjects, 1]
        mask: timepoints to evaluate, [nb_timpoints, nb_subjects, 1]
    Returns:
        cross-entropy loss
    """
    assert isinstance(pred, torch.Tensor)
    assert isinstance(true, np.ndarray) and isinstance(mask, np.ndarray)
    nb_subjects = true.shape[1]

    pred = pred.reshape(pred.size(0) * pred.size(1), -1)
    mask = mask.reshape(-1, 1)

    o_true = pred.new_tensor(true.reshape(-1, 1)[mask], dtype=torch.long)
    o_pred = pred[pred.new_tensor(
        mask.squeeze(1).astype(np.uint8), dtype=torch.uint8)]

    return torch.nn.functional.cross_entropy(
        o_pred, o_true, reduction='sum') / nb_subjects


def mae_loss(pred, true, mask):
    """
    Calculate mean absolute error (MAE)
    Args:
        pred: predicted values, [nb_timpoints, nb_subjects, nb_features]
        true: true values, [nb_timpoints, nb_subjects, nb_features]
        mask: values to evaluate, [nb_timpoints, nb_subjects, nb_features]
    Returns:
        MAE loss
    """
    assert isinstance(pred, torch.Tensor)
    assert isinstance(true, np.ndarray) and isinstance(mask, np.ndarray)
    nb_subjects = true.shape[1]

    invalid = ~mask
    true[invalid] = 0
    indices = pred.new_tensor(invalid.astype(np.uint8), dtype=torch.uint8)
    assert pred.shape == indices.shape
    pred[indices] = 0

    return torch.nn.functional.l1_loss(
        pred, pred.new(true), reduction='sum') / nb_subjects


def to_cat_seq(labels):
    """
    Return one-hot representation of a sequence of class labels
    Args:
        labels: [nb_subjects, nb_timpoints]
    Returns:
        [nb_subjects, nb_timpoints, nb_classes]
    """
    return np.asarray([misc.to_categorical(c, 3) for c in labels])


def train_1epoch(args, model, dataset, optimizer, device):
    """
    Train an recurrent model for 1 epoch
    Args:
        args: include training hyperparametres and input/output paths
        model: pytorch model object
        dataset: training data
        optimizer: optimizer
    Returns:
        cross-entropy loss of epoch
        mean absolute error (MAE) loss of epoch
    """
    model.train()
    total_kl = total_nll = 0

    prev_data = torch.tensor(np.zeros((dataset.batch_size, model.embedder._input_dim, 4)), dtype=torch.float64, device=device)
    for batch in dataset:
        for tp, cat, val, cat_m, val_m, t_cat, t_val in zip(batch['tp'], batch['cat'], batch['val'], batch['cat_msk'], batch['val_msk'], batch['true_cat'], batch['true_val']):
            # if len(tp) == 1:
            #     continue
            # if ~cat_m.all():
            #     continue
            np.nan_to_num(cat, nan=0, copy=False)
            np.nan_to_num(val, nan=0, copy=False)
            np.nan_to_num(t_cat, nan=0, copy=False)
            np.nan_to_num(t_val, nan=0, copy=False)

            batch_size, feat_cnt = val.shape
            if batch_size != 128:
                continue

            
            assert feat_cnt == 22
            assert batch_size == 128

            optimizer.zero_grad()
            cat, val, cat_m, val_m = (to_cat_seq(cat), val, cat_m, val_m)

            
            data = np.zeros((batch_size, model.embedder._input_dim, 4))

            data[:,0,:3] = cat.squeeze()
            data[:,1:,0] = val.squeeze()
            data[:,0,3] = cat_m.squeeze()
            data[:,1:,3] = val_m.squeeze()
            
            data_np = data
            
            data = torch.tensor(data, dtype=torch.float64, device=device)
            (dec_mean, dec_logvar), enc_samples, (enc_mean, enc_logvar) = model(data, prev_data)
            
            t_cat = to_cat_seq(t_cat).squeeze()
            t_cat = torch.tensor(t_cat, dtype=torch.float64, device=device)
            t_val = torch.tensor(t_val, dtype=torch.float64, device=device)

            t_cat = torch.nan_to_num(t_cat, nan=0)    # ravi: TODO: check this
            t_val = torch.nan_to_num(t_val, nan=0)

            cat_m = torch.tensor(cat_m, dtype=torch.float64, device=device)
            val_m = torch.tensor(val_m, dtype=torch.float64, device=device)

            kl = kl_divergence((enc_mean, enc_logvar)).mean()

            nll = negative_log_likelihood((t_val, t_cat), dec_mean, dec_logvar, alpha=1, mask=(val_m, cat_m))

            # mask_cat = batch['cat_msk'][1:]
            # # assert mask_cat.sum() > 0

            # ent = ent_loss(pred_cat, batch['true_cat'][1:], mask_cat)
            # mae = mae_loss(pred_val, batch['true_val'][1:], batch['val_msk'][1:])
            total_loss = 0*kl + args.w_ent * nll

            total_loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0)
            optimizer.step()

            batch_size = cat_m.shape[0]
            total_kl += kl.item() 
            total_nll += nll.item()
            
            prev_data = data

    return total_kl / len(dataset.subjects), total_nll / len(dataset.subjects)


def save_config(args, config_path):
    """
    Save training configuration as json file
    Args:
        args: include training hyperparametres and input/output paths
        config_path: path of output json file
    Returns:
        None
    """
    with open(config_path, 'w') as fhandler:
        print(json.dumps(vars(args), sort_keys=True), file=fhandler)


def train(args):
    """
    Train an recurrent model
    Args:
        args: include training hyperparametres and input/output paths
    Returns:
        None
    """
    log = print if args.verbose else lambda *x, **i: None

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.data, 'rb') as fhandler:
        data = pickle.load(fhandler)
    nb_measures = len(data['train'].value_fields())


    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # model_class = MODEL_DICT[args.model]
    # model = model_class(
    #     nb_classes=3,
    #     nb_measures=nb_measures,
    #     nb_layers=args.nb_layers,
    #     h_size=args.h_size,
    #     h_drop=args.h_drop,
    #     i_drop=args.i_drop)
    model = Transformer()
    if args.load is not None:
        model = torch.load(args.load)

    model = model.double()
    setattr(model, 'min', data['min'])
    setattr(model, 'max', data['max'])

    model.to(device)
    log(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start = time.time()
    try:
        for i in range(args.epochs):
            loss = train_1epoch(args, model, data['train'], optimizer, device)
            log_info = (i + 1, args.epochs, misc.time_from(start)) + loss
            log('%d/%d %s KLD %.3f, NLL %.3f' % log_info)
    except KeyboardInterrupt:
        print('Early exit')

    torch.save(model, args.out)
    save_config(args, '%s.json' % args.out)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', required=True)

    parser.add_argument('--data', required=True)
    parser.add_argument('--out', '-o', required=True)

    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--w_ent', type=float, default=1.)

    parser.add_argument("--load", required=False, default=None)
    parser.add_argument('--nb_layers', type=int, default=1)
    parser.add_argument('--h_size', type=int, default=512)
    parser.add_argument('--i_drop', type=float, default=.0)
    parser.add_argument('--h_drop', type=float, default=.0)
    parser.add_argument('--weight_decay', type=float, default=.0)

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    train(get_args())
