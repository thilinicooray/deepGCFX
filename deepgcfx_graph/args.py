import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GVAE Arguments.')
    parser.add_argument('--DS', dest='DS', help='Dataset')
    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
                        help='')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128,
                        help='')
    parser.add_argument('--num_epochs', dest='num_epochs', help='number of epochs', type=int)
    parser.add_argument('--model_name', dest='model_name', help='Name to save the model')

    return parser.parse_args()

