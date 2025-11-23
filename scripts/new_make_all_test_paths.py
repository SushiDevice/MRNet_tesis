#!/usr/bin/env python3.6
"""Creates a csv containing paths for a dataset split (default: validation)
to be used as an argument to src/predict.py.

Usage:
  make_all_valid_paths.py <data_dir> <output_dir> [--split=<name>]
  make_all_valid_paths.py (-h | --help)

General options:
  -h --help          Show this screen.

Arguments:
  <data_dir>         Path to a directory where the data lives e.g. 'MRNet-v1.0'
  <output_dir>       Directory where paths are saved as a csv file (with no header)
                     e.g. 'out_dir'

Options:
  --split=<name>     Split folder under <data_dir> to read cases from.
                     Examples: 'valid' (por defecto), 'test_split1', 'train_split1'
                     [default: valid]
"""

import os
import sys
import csv
from typing import List
from docopt import docopt


def _list_case_basenames(dir_path: str) -> List[str]:
    """Return sorted list of case basenames (without extension) found in dir_path.

    Sorting tries numeric order if filenames are numeric, otherwise falls back to lexicographic.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f'Directory not found: {dir_path}')

    files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
    basenames = [os.path.splitext(f)[0] for f in files]

    def _to_int_or_str(x: str):
        try:
            return int(x)
        except ValueError:
            return x

    return sorted(basenames, key=_to_int_or_str)


def main(data_dir, output_dir, split):
    base_split_path = f'{data_dir}/{split}'
    planes = ['sagittal', 'coronal', 'axial']  # Order expected by predict.py

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # nombre de salida en función del split, p.ej. out/test_split1_paths.csv
    output_file = f'{output_dir}/{split}_paths.csv'

    if os.path.exists(output_file):
        os.rename(output_file, f'{output_file}.back')
        print(f'!! {output_file} already exists, renamed to {output_file}.bak')

    print(f'Generating a list of paths for split="{split}"...')
    print(f'Reading cases from: {base_split_path}')
    print(f'Paths will be saved to: {output_file}')

    # Detectar casos dinámicamente a partir del plano sagittal
    sagittal_dir = os.path.join(base_split_path, 'sagittal')
    case_basenames = _list_case_basenames(sagittal_dir)

    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        for case_base in case_basenames:
            for plane in planes:
                case_path = f'{base_split_path}/{plane}/{case_base}.npy'
                writer.writerow([case_path])


if __name__ == '__main__':
    arguments = docopt(__doc__)

    print('Parsing arguments...')

    main(arguments['<data_dir>'],
         arguments['<output_dir>'],
         arguments['--split'])
