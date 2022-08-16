import torch
import argparse


parser = argparse.ArgumentParser(description='mri_pituitary')

# reproducibility options
parser.add_argument('--seed', type=int, default=42, metavar='seed', help='random seed (default: 42)')

# data options
parser.add_argument('--n-train', type=int, default=80, metavar='n_train',
                    help='number of training + validation samples (default: 80)')
parser.add_argument('--p-train', type=float, default=0.8, metavar='p_train',
                    help='percentage of training samples (default: 0.8)')

# training parameters
parser.add_argument('--max-epochs', type=int, default=50, metavar='max_epochs',
                    help='maximum of epochs to train (default: 50)')

#%% ================================================================================================================ %%#



