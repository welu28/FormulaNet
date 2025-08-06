import os
import sys
import argparse
import pickle
import random

import torch

from holstep_parser import graph_from_hol_stmt
from holstep_parser import tree_from_hol_stmt


def generate_dataset(path, output, partition, converter, files=None, batch_size=None):
    '''Generate dataset at given path in batches

    Parameters
    ----------
    path : str
        Path to the source
    output : str
        Path to the destination
    partition : int
        Number of the partition for this dataset (i.e. # of files)
    batch_size : int or None
        Number of files to process per batch. If None, processes all at once.
    '''
    digits = len(str(partition))
    outputs = [[] for _ in range(partition)]

    if files is None:
        files = os.listdir(path)

    total_files = len(files)
    batch_size = batch_size or total_files  # default all files at once

    for batch_start in range(0, total_files, batch_size):
        batch_files = files[batch_start: batch_start + batch_size]
        print(f'Processing batch {batch_start // batch_size + 1} / {(total_files + batch_size - 1) // batch_size} with {len(batch_files)} files.')

        for i, fname in enumerate(batch_files):
            fpath = os.path.join(path, fname)
            print('Processing file {}/{} at {}.'.format(i + 1 + batch_start, total_files, fpath))
            with open(fpath, 'r') as f:
                next(f)
                conj_symbol = next(f)
                conj_token = next(f)
                assert conj_symbol[0] == 'C'
                conjecture = converter(conj_symbol[2:], conj_token[2:])
                for line in f:
                    if line and line[0] in '+-':
                        statement = converter(line[2:], next(f)[2:])
                        flag = 1 if line[0] == '+' else 0
                        record = flag, conjecture, statement
                        outputs[random.randint(0, partition - 1)].append(record)

        # After each batch, save the current accumulated data and clear outputs
        for i, data in enumerate(outputs):
            with open(
                    os.path.join(output, 'holstep' + format(i, "0{}d".format(digits))),
                    'ab') as f:  # use 'ab' to append batches
                print('Saving batch to file {}/{}'.format(i + 1, partition))
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            outputs[i] = []  # clear for next batch


if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    parser = argparse.ArgumentParser(
        description='Generate graph repr dataset from HolStep with batch processing')

    parser.add_argument('path', type=str, help='Path to the root of HolStep dataset')
    parser.add_argument('output', type=str, help='Output folder')
    parser.add_argument(
        '--train_partition',
        '-train',
        type=int,
        help='Number of the partition of the training dataset. Default=200',
        default=200)
    parser.add_argument(
        '--test_partition',
        '--test',
        type=int,
        help='Number of the partition of the testing dataset. Default=20',
        default=20)
    parser.add_argument(
        '--valid_partition',
        '--valid',
        type=int,
        help='Number of the partition of the validation dataset. Default=20',
        default=20)
    parser.add_argument(
        '--format',
        type=str,
        default='graph',
        help='Format of the representation. Either tree or graph (default).')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Number of files to process per batch (default all at once).')

    args = parser.parse_args()

    format_choice = {
        'graph': lambda x, y: graph_from_hol_stmt(x, y),
        'tree': lambda x, y: tree_from_hol_stmt(x, y)
    }

    assert os.path.isdir(args.output), 'Data path must be a folder'
    assert os.path.isdir(args.path), 'Saving path must be a folder'
    train_output = os.path.join(args.output, 'train')
    test_output = os.path.join(args.output, 'test')
    valid_output = os.path.join(args.output, 'valid')
    train_path = os.path.join(args.path, 'train')
    test_path = os.path.join(args.path, 'test')
    valid_path = os.path.join(args.path, 'valid')

    print(train_path, train_output)
    if not os.path.exists(train_output):
        os.mkdir(train_output)
    if not os.path.exists(test_output):
        os.mkdir(test_output)
    if not os.path.exists(valid_output):
        os.mkdir(valid_output)

    files = os.listdir(train_path)
    valid_files = random.sample(files, int(len(files) * 0.07 + 0.5))
    train_files = [x for x in files if x not in valid_files]

    print(valid_files)
    print(train_files)

    generate_dataset(train_path, train_output, args.train_partition,
                     format_choice[args.format], train_files, batch_size=args.batch_size)
    generate_dataset(test_path, test_output, args.test_partition,
                     format_choice[args.format], batch_size=args.batch_size)
    generate_dataset(train_path, valid_output, args.valid_partition,
                 format_choice[args.format], valid_files, batch_size=args.batch_size)
