import os
import argparse
import logging
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-dir', type=str, default='./data')
    parser.add_argument('--model-dir', type=str, default='./model/final')
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--nodes', type=str, default='features.csv')
    parser.add_argument('--target-ntype', type=str, default='TransactionID')
    parser.add_argument('--edges', type=str, default='relation*')
    parser.add_argument('--labels', type=str, default='tags.csv')
    parser.add_argument('--new-accounts', type=str, default='test.csv')
    parser.add_argument('--compute-metrics', type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        default=True, help='compute evaluation metrics after training')
    parser.add_argument('--threshold', type=float, default=0,
                        help='threshold for making predictions, default : argmax')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='number of GPUs to use, must be >= 1')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n-epochs', type=int, default=700)
    parser.add_argument('--n-hidden', type=int, default=16, help='number of hidden units')
    parser.add_argument('--n-layers', type=int, default=3, help='number of hidden layers')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight for L2 loss')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout probability, for gat only features')
    parser.add_argument('--embedding-size', type=int, default=360, help='embedding size for node embedding')
    return parser.parse_known_args()[0]


def get_device(num_gpus):
    """
    Validate CUDA availability and return device.
    Raises RuntimeError if no GPU is found.

    :param num_gpus: number of GPUs requested
    :return: torch.device
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. A GPU is required to run this code. "
            "Please check your environment and ensure a CUDA-compatible GPU is present."
        )
    available = torch.cuda.device_count()
    if num_gpus > available:
        raise RuntimeError(
            "Requested {} GPU(s) but only {} available.".format(num_gpus, available)
        )
    device = torch.device('cuda')
    print("Using GPU: {} - {}".format(torch.cuda.current_device(), torch.cuda.get_device_name(0)))
    return device


def get_logger(name):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger.setLevel(logging.INFO)
    return logger
