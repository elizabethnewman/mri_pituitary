import argparse


def setup_parser():
    parser = argparse.ArgumentParser(description='U-Net for MRI Pituitary Segementation')

    # reproducibility options
    parser.add_argument('--seed', type=int, default=42, metavar='seed', help='random seed (default: 42)')

    # data options
    parser.add_argument('--n-train', type=int, default=80, metavar='n_train',
                        help='number of training + validation samples (default: 80)')
    parser.add_argument('--p-train', type=float, default=0.8, metavar='p_train',
                        help='percentage of training samples (default: 0.8)')

    # data setup
    parser.add_argument('--normalize', type=str, default='image_standardization', help='method to normalize data')
    # TODO: add morphology options

    # training parameters
    parser.add_argument('--max-epochs', type=int, default=50, metavar='max_epochs',
                        help='maximum of epochs to train (default: 50)')

    # network parameters
    parser.add_argument('--in-channels', type=int, default=1, metavar='n_channels',
                        help='number of input channels (default: 1)')
    parser.add_argument('--width', type=int, default=8, help='width of U-Net (default: 8)')
    parser.add_argument('--n-classes', type=int, default=4, metavar='n_channels',
                        help='number of classes (default: 4)')

    # optimizer parameters
    parser.add_argument('--sgd', action='store_true', help='optimize with SGD (default: L-BFGS')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--regularize', action='store_true', help='add Tikhonov regularization')
    parser.add_argument('--alpha', type=float, default=1e-4, help='regularization parameter')
    parser.add_argument('--line-search', type=str, default='strong_wolfe', help='line search function for L-BFGS')
    parser.add_argument('--verbose', action='store_true', help='print training statistics')
    # TODO: add decaying factors (decrease lr and/or alpha)

    # visualizations
    parser.add_argument('--plot', action='store_true', help='show plots while training')

    # saving
    parser.add_argument('--save', action='store_true', help='save network parameters')

    return parser


def setup_filename(args):
    filename = 'mri'
    for a in args.__dict__:
        filename += '--' + a + '-' + str(getattr(args, a))

    filename += '.pt'
    return filename

