import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

# ============================================================
# Checkpoint configuration - change this value as needed
CHECKPOINT_EVERY = 100       # save checkpoint every N epochs
CHECKPOINT_DIR = './model/checkpoint'  # fixed directory, overwritten each time
# ============================================================

import torch as th
import torch_geometric
print('PyTorch Geometric version: {}'.format(torch_geometric.__version__))
import numpy as np
import pandas as pd
import time
import pickle
import copy

from sklearn.metrics import confusion_matrix
from gnn.estimator_fns import *
from gnn.graph_utils import *
from gnn.data import *
from gnn.utils import *
from gnn.pytorch_model import *


def initial_record():
    if os.path.exists('./output/results.txt'):
        os.remove('./output/results.txt')
    with open('./output/results.txt', 'w') as f:
        f.write("Epoch,Time(s),Loss,F1\n")


def normalize(feature_matrix):
    mean = th.mean(feature_matrix, axis=0)
    stdev = th.sqrt(th.sum((feature_matrix - mean) ** 2, axis=0) / feature_matrix.shape[0])
    return mean, stdev, (feature_matrix - mean) / stdev


def save_checkpoint(model, optim, epoch, loss, model_dir):
    """
    Save model checkpoint to disk including optimizer state for resume training.

    :param model: current best model
    :param optim: optimizer
    :param epoch: current epoch number
    :param loss: current best loss value
    :param model_dir: directory to save checkpoint
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    checkpoint_path = os.path.join(model_dir, 'checkpoint_best.pth')
    th.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'best_loss': loss,
    }, checkpoint_path)
    print("Checkpoint saved at epoch {:05d}, loss {:.4f} -> {}".format(
        epoch, loss, checkpoint_path))


def load_checkpoint(model, optim, model_dir, device):
    """
    Load checkpoint from disk to resume training.

    :param model: initialized model (same architecture)
    :param optim: initialized optimizer
    :param model_dir: directory containing checkpoint
    :param device: torch device
    :return: (model, optim, start_epoch, best_loss) or None if no checkpoint found
    """
    checkpoint_path = os.path.join(model_dir, 'checkpoint_best.pth')
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found at {}, starting from scratch".format(checkpoint_path))
        return model, optim, 0, 1.0

    checkpoint = th.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']
    print("Resumed from checkpoint: epoch {:05d}, best_loss {:.4f}".format(
        checkpoint['epoch'], best_loss))
    return model, optim, start_epoch, best_loss


def train_fg(model, optim, loss, features, labels, train_g, test_g, test_mask,
             device, n_epochs, thresh, model_dir, start_epoch=0, best_loss=1.0, compute_metrics=True):
    """
    Full graph version of RGCN training.
    Supports resume from checkpoint via start_epoch and best_loss.
    """
    duration = []
    # initialize best_model from current model (important for resume case)
    # prevents best_model being None if no epoch improves loss after resume
    best_model = copy.deepcopy(model)

    for epoch in range(start_epoch, n_epochs):
        tic = time.time()
        loss_val = 0.

        pred = model(train_g, features.to(device))
        l = loss(pred, labels)

        optim.zero_grad()
        l.backward()
        optim.step()

        loss_val += l

        duration.append(time.time() - tic)
        metric = evaluate(model, train_g, features, labels, device)
        print("Epoch {:05d}, Time(s) {:.4f}, Loss {:.4f}, F1 {:.4f} ".format(
            epoch, np.mean(duration), loss_val, metric))

        epoch_result = "{:05d},{:.4f},{:.4f},{:.4f}\n".format(
            epoch, np.mean(duration), loss_val, metric)
        with open('./output/results.txt', 'a+') as f:
            f.write(epoch_result)

        # update best model whenever loss improves (runs throughout all epochs)
        # use .item() to keep best_loss as float, not tensor
        if loss_val < best_loss:
            best_loss = loss_val.item()
            best_model = copy.deepcopy(model)  # replaces previous, always 1 copy in RAM

        # save checkpoint every CHECKPOINT_EVERY epochs to fixed directory
        # always saves the best model seen so far, overwrites previous checkpoint
        if (epoch + 1) % CHECKPOINT_EVERY == 0 and best_model is not None:
            save_checkpoint(best_model, optim, epoch, best_loss, CHECKPOINT_DIR)

    class_preds, pred_proba = get_model_class_predictions(best_model,
                                                           test_g,
                                                           features,
                                                           labels,
                                                           device,
                                                           threshold=thresh)
    if compute_metrics:
        acc, f1, p, r, roc, pr, ap, cm = get_metrics(
            class_preds, pred_proba, labels.cpu().numpy(), test_mask.cpu().numpy(), './output/')
        print("Metrics")
        print("""Confusion Matrix:
                        {}
                        f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, acc: {:.4f}, roc: {:.4f}, pr: {:.4f}, ap: {:.4f}
                     """.format(cm, f1, p, r, acc, roc, pr, ap))

    return best_model, class_preds, pred_proba


def get_f1_score(y_true, y_pred):
    """
    Only works for binary case.
    tn, fp, fn, tp = cf_m[0,0], cf_m[0,1], cf_m[1,0], cf_m[1,1]
    """
    cf_m = confusion_matrix(y_true, y_pred)
    precision = cf_m[1, 1] / (cf_m[1, 1] + cf_m[0, 1] + 10e-5)
    recall = cf_m[1, 1] / (cf_m[1, 1] + cf_m[1, 0])
    f1 = 2 * (precision * recall) / (precision + recall + 10e-5)
    return precision, recall, f1


def evaluate(model, g, features, labels, device):
    """Compute F1 value in a binary classification case"""
    with th.no_grad():  # disable gradient computation, saves VRAM and speeds up evaluate
        preds = model(g, features.to(device))
        preds = th.argmax(preds, axis=1).cpu().numpy()
    precision, recall, f1 = get_f1_score(labels.cpu().numpy(), preds)
    return f1


def get_model_class_predictions(model, g, features, labels, device, threshold=None):
    unnormalized_preds = model(g, features.to(device))
    pred_proba = th.softmax(unnormalized_preds, dim=-1)
    if not threshold:
        return (unnormalized_preds.argmax(axis=1).detach().cpu().numpy(),
                pred_proba[:, 1].detach().cpu().numpy())
    return (np.where(pred_proba.detach().cpu().numpy() > threshold, 1, 0),
            pred_proba[:, 1].detach().cpu().numpy())


def save_model(g, model, model_dir, id_to_node, mean, stdev):
    # TẠO THƯ MỤC NẾU CHƯA TỒN TẠI (Sửa lỗi OSError)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save best model parameters
    th.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

    # Save graph metadata for inference
    etype_list = g.edge_types
    ntype_cnt = {ntype: g[ntype].num_nodes for ntype in g.node_types}

    with open(os.path.join(model_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump({'etypes': etype_list,
                     'ntype_cnt': ntype_cnt,
                     'feat_mean': mean,
                     'feat_std': stdev}, f)

    # Save original IDs and trained embeddings for non-target node types
    for ntype, mapping in id_to_node.items():
        if ntype == 'target':
            continue

        old_id_list, node_id_list = [], []
        for old_id, node_id in mapping.items():
            old_id_list.append(old_id)
            node_id_list.append(node_id)

        node_feats = model.embed[ntype].detach().cpu().numpy()
        num_nodes = node_feats.shape[0]
        num_feats = node_feats.shape[1]

        node_ids_df = pd.DataFrame({'~label': [ntype] * num_nodes})
        node_ids_df['~id_tmp'] = old_id_list
        node_ids_df['~id'] = node_ids_df['~label'] + '-' + node_ids_df['~id_tmp']
        node_ids_df['node_id'] = node_id_list

        cols = {'val' + str(i + 1) + ':Double': node_feats[:, i] for i in range(num_feats)}
        node_feats_df = pd.DataFrame(cols)

        node_id_feats_df = node_ids_df.merge(node_feats_df, left_on='node_id', right_on=node_feats_df.index)
        node_id_feats_df = node_id_feats_df.drop(['~id_tmp', 'node_id'], axis=1)

        node_id_feats_df.to_csv(os.path.join(model_dir, ntype + '.csv'),
                                index=False, header=True, encoding='utf-8')


def get_model(ntype_dict, etypes, hyperparams, in_feats, n_classes, device):
    model = HeteroRGCN(ntype_dict, etypes, in_feats, hyperparams['n_hidden'],
                       n_classes, hyperparams['n_layers'], in_feats)
    model = model.to(device)
    return model


if __name__ == '__main__':
    print('numpy version:{} PyTorch version:{} PyG version:{}'.format(
        np.__version__, th.__version__, torch_geometric.__version__))

    args = parse_args()
    print(args)

    # validate GPU availability early
    device = get_device(args.num_gpus)

    args.edges = get_edgelists('relation*', args.training_dir)

    g, features, target_id_to_node, id_to_node = construct_graph(
        args.training_dir, args.edges, args.nodes, args.target_ntype)

    mean, stdev, features = normalize(th.from_numpy(features))
    print('feature mean shape:{}, std shape:{}'.format(mean.shape, stdev.shape))

    # update normalized features on graph
    g['target'].x = features.to(device)

    print("Getting labels")
    n_nodes = g['target'].num_nodes

    labels, _, test_mask = get_labels(target_id_to_node,
                                      n_nodes,
                                      args.target_ntype,
                                      os.path.join(args.training_dir, args.labels),
                                      os.path.join(args.training_dir, args.new_accounts))
    print("Got labels")

    labels = th.from_numpy(labels).float()
    test_mask = th.from_numpy(test_mask).float()

    # count total nodes and edges across all types
    n_nodes = th.sum(th.tensor([g[ntype].num_nodes for ntype in g.node_types]))
    n_edges = th.sum(th.tensor([g[src, etype, dst].edge_index.shape[1]
                                 for src, etype, dst in g.edge_types]))

    print("""----Data statistics------
                #Nodes: {}
                #Edges: {}
                #Features Shape: {}
                #Labeled Test samples: {}""".format(
        n_nodes, n_edges, features.shape, test_mask.sum()))

    print("Initializing Model")
    in_feats = features.shape[1]
    n_classes = 2

    ntype_dict = {ntype: g[ntype].num_nodes for ntype in g.node_types}

    # edge type strings for model (etype only, not full canonical tuple)
    etypes = [etype for _, etype, _ in g.edge_types]

    model = get_model(ntype_dict, etypes, vars(args), in_feats, n_classes, device)
    print("Initialized Model")

    features = features.to(device)
    labels = labels.long().to(device)
    test_mask = test_mask.to(device)

    loss = th.nn.CrossEntropyLoss()
    optim = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Starting Model training")

    # resume from checkpoint if available (always from fixed CHECKPOINT_DIR)
    model, optim, start_epoch, best_loss = load_checkpoint(model, optim, CHECKPOINT_DIR, device)

    if start_epoch == 0:
        initial_record()
    else:
        print("Resuming training from epoch {:05d}".format(start_epoch))

    # pass model_dir, start_epoch, best_loss into train_fg for resume support
    best_model, class_preds, pred_proba = train_fg(model, optim, loss, features, labels, g, g,
                                                   test_mask, device, args.n_epochs,
                                                   args.threshold, args.model_dir,
                                                   start_epoch=start_epoch,
                                                   best_loss=best_loss,
                                                   compute_metrics=args.compute_metrics)
    print("Finished Model training")

    print("Saving best model")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # save best_model instead of model (epoch cuoi)
    save_model(g, best_model, args.model_dir, id_to_node, mean, stdev)
    print("Model and metadata saved")
