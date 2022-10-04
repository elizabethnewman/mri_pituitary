import torch
import argparse
from mri_pituitary.utils import seed_everything
from mri_pituitary.my_parser import setup_parser, setup_filename

# this script sets up the optimizer

parser = setup_parser()
args = parser.parse_known_args()
args = args[0]
print(args)

filename = setup_filename(args)
print(filename)


#%% ================================================================================================================ %%#
seed_everything(args.seed)

# get device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# assume data has name images, masks

