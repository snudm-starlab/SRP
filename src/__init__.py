"""
Starlab Transformer Compression with SRP (Selectively Regularized Pruning)

Author: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
        U Kang (ukang@snu.ac.kr), Seoul National University

Version : 1.0
Date : Nov 29, 2022
Main Contact: Hyojin Jeon

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.
This code is mainly based on the [GitHub Repository]
[GitHub Repository]: https://github.com/facebookresearch/fairseq
"""

import importlib
import os

# recursively and automatically import any Python files in the current directory
CUR_DIR = os.path.dirname(__file__)
if CUR_DIR == '':
    CUR_DIR = '.'
tar_dirs = ['criterions', 'optim']

def import_py_files(base_dir, _dir):
    """Import all the python files in the directory"""
    dir_path = os.path.join(base_dir, _dir)

    for file in os.listdir(dir_path):
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py"))
        ):
            mod_name = file[: file.find(".py")] if file.endswith(".py") else file
            path_name = _dir.replace('/', '.')
            _ = importlib.import_module(f'{__name__}.{path_name}.{mod_name}')

for tar_dir in tar_dirs:
    import_py_files(CUR_DIR, tar_dir)
