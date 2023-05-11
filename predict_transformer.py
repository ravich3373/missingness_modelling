#!/usr/bin/env python
# encoding: utf-8
# Written by Minh Nguyen and CBIG under MIT license:
# https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
from __future__ import print_function
import argparse
import pickle

import numpy as np
import torch

import misc as misc
from tqdm import tqdm


def predict_subject(model, cat_seq, value_seq, time_seq, device):
    """
    Predict Alzheimer’s disease progression for a subject
    Args:
        model: trained pytorch model
        cat_seq: sequence of diagnosis [nb_input_timpoints, nb_classes]
        value_seq: sequence of other features [nb_input_timpoints, nb_features]
        time_seq: months from baseline [nb_output_timpoints, nb_features]
    nb_input_timpoints <= nb_output_timpoints
    Returns:
        out_cat: predicted diagnosis
        out_val: predicted features
    """
    in_val = np.full((len(time_seq), ) + value_seq.shape[1:], np.nan)
    in_val[:len(value_seq)] = value_seq

    in_cat = np.full((len(time_seq), ) + cat_seq.shape[1:], np.nan)
    in_cat[:len(cat_seq)] = cat_seq

    in_val = np.moveaxis(in_val, [0, 1], [1, 0])
    in_cat = np.moveaxis(in_cat, [0, 1], [1, 0])

    batch_size, seq_len, _ = in_val.shape
    
    cat_m = (np.isnan(in_cat).sum(axis=-1) > 2)[:,:,None].astype(int)
    val_m = np.isnan(in_val).astype(int)

    in_cat = np.nan_to_num(in_cat, 0)
    in_val = np.nan_to_num(in_val, 0)

    features_to_concatenate = list(map(torch.tensor, [in_cat, in_val, cat_m, val_m]))
        
    data = torch.cat(features_to_concatenate, dim=-1).double().to(device)
    # data = np.zeros((batch_size+1, model.embedder._input_dim, 4))

    # cat_m = (np.isnan(in_cat).squeeze().sum(axis=1) > 2).astype(int)
    # val_m = np.isnan(in_val).astype(int)

    # 

    # data[1:,0,:3] = in_cat.squeeze()
    # data[1:,1:,0] = in_val.squeeze()
    # data[1:,0,3] = cat_m.squeeze()
    # data[1:,1:,3] = val_m.squeeze()
    
    serial = False
    if serial:
        curr_data = data[1:]
        prev_sample = torch.tensor(data[:1]).double().to(model.embedder._embedding_weights.device)

        cat_preds = []
        cont_preds = []
        with torch.no_grad():
            for curr_sample in curr_data:
                curr_sample = torch.tensor(np.expand_dims(curr_sample, 0)).double().to(model.embedder._embedding_weights.device)
                (dec_mean, dec_logvar), enc_samples, (enc_mean, enc_logvar) = model(curr_sample, prev_sample)

                prev_sample = dec_mean[:,:,:4]
                cat_preds.append(list(dec_mean[0,0,:3].detach().cpu().softmax(-1)))
                cont_preds.append(list(dec_mean[0,1:,0].detach().cpu().sigmoid()))
    else:
        curr_data = data
        prev_data = None
        with torch.no_grad():
            (dec_mean, dec_logvar), enc_samples, (enc_mean, enc_logvar) = model(curr_data, prev_data)
        
        cont_preds = (dec_mean[:,:,3:25]*dec_mean[:,:,-22:]).sigmoid()
        cat_preds = (dec_mean[:,:,:3]* dec_mean[:,:,25][:,:,None]).softmax(-1)

        # cat_preds = dec_mean[:,0,:3].detach().cpu().softmax(-1)
        # cont_preds = dec_mean[:,1:,0].detach().cpu().sigmoid()

    out_cat = np.array(cat_preds.detach().cpu())
    out_val = np.array(cont_preds.detach().cpu())

    # out_cat = np.expand_dims(out_cat, 1)
    # out_val = np.expand_dims(out_val, 1)

    out_val = np.moveaxis(out_val, [0, 1], [1, 0])
    out_cat = np.moveaxis(out_cat, [0, 1], [1, 0])

    assert out_cat.shape[1] == out_val.shape[1] == 1

    return out_cat, out_val


def predict(model, dataset, pred_start, duration, baseline, device):
    """
    Predict Alzheimer’s disease progression using a trained model
    Args:
        model: trained pytorch model
        dataset: test data
        pred_start (dictionary): the date at which prediction begins
        duration (dictionary): how many months into the future to predict
        baseline (dictionary): the baseline date
    Returns:
        dictionary which contains the following key/value pairs:
            subjects: list of subject IDs
            DX: list of diagnosis prediction for each subject
            ADAS13: list of ADAS13 prediction for each subject
            Ventricles: list of ventricular volume prediction for each subject
    """
    model.eval()
    ret = {'subjects': dataset.subjects}
    ret['DX'] = []  # 1. likelihood of NL, MCI, and Dementia
    ret['ADAS13'] = []  # 2. (best guess, upper and lower bounds on 50% CI)
    ret['Ventricles'] = []  # 3. (best guess, upper and lower bounds on 50% CI)
    ret['dates'] = misc.make_date_col(
        [pred_start[s] for s in dataset.subjects], duration)

    col = ['ADAS13', 'Ventricles', 'ICV']
    indices = misc.get_index(list(dataset.value_fields()), col)
    mean = model.mean[col].values.reshape(1, -1)
    std = model.stds[col].values.reshape(1, -1)

    for data in tqdm(dataset):
        rid = data['rid']
        all_tp = data['tp'].squeeze(axis=1)
        start = misc.month_between(pred_start[rid], baseline[rid])
        assert np.all(all_tp == np.arange(len(all_tp)))
        mask = all_tp < start
        itime = np.arange(start + duration)
        icat = np.asarray(
            [misc.to_categorical(c, 3) for c in data['cat'][mask]])
        ival = data['val'][:, None, :][mask]

        ocat, oval = predict_subject(model, icat, ival, itime, device)
        oval = oval[-duration:, 0, indices] * std + mean

        ret['DX'].append(ocat[-duration:, 0, :])
        ret['ADAS13'].append(misc.add_ci_col(oval[:, 0], 1, 0, 85))
        ret['Ventricles'].append(
            misc.add_ci_col(oval[:, 1] / oval[:, 2], 5e-4, 0, 1))

    return ret


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', '-o', required=True)

    return parser.parse_args()


def main(args):
    """
    Predict Alzheimer’s disease progression using a trained model
    Save prediction as a csv file
    Args:
        args: includes model path, input/output paths
    Returns:
        None
    """
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(args.checkpoint)
    model.to(device)

    with open(args.data, 'rb') as fhandler:
        data = pickle.load(fhandler)

    prediction = predict(model, data['test'], data['pred_start'],
                         data['duration'], data['baseline'], device)
    misc.build_pred_frame(prediction, args.out)


if __name__ == '__main__':
    main(get_args())
