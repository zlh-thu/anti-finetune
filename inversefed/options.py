"""Parser options."""

import argparse

def attack_options():
    """Construct the central argument parser, filled with useful defaults."""
    parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')

    # Central:
    #parser.add_argument('--model', default='ConvNet', type=str, help='Vision model.')
    #parser.add_argument('--dataset', default='CIFAR10', type=str)


    #parser.add_argument('--image_path', default='images/', type=str)
    #parser.add_argument('--model_path', default='models/', type=str)
    #parser.add_argument('--table_path', default='tables/', type=str)
    #parser.add_argument('--data_path', default='~/data', type=str)

    # Debugging:
    #parser.add_argument('--name', default='iv', type=str, help='Name tag for the result table and model.')
    #parser.add_argument('--deterministic', action='store_true', help='Disable CUDNN non-determinism.')
    parser.add_argument('--dryrun', action='store_true', help='Run everything for just one step to test functionality.')
    return parser
