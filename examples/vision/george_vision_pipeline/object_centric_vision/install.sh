#! /bin/bash
#
# install.sh
# Copyright (C) 2023 Jiayuan Mao <maojiayuan@gmail.com>
#
# Distributed under terms of the MIT license.
#
#
# Iteratively parse the arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            echo "Usage: $0 [-h|--help] [--install-pytorch]"
            exit 0
            ;;
        --install-pytorch)
            echo 'Install pytorch'
            conda install pytorch torchvision -c pytorch
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
  done

echo "Install neovim python bindings"
conda install neovim
echo "Install python LSP"
pip install python-lsp-server
echo "Install scipy scikit-learn pandas matplotlib numpy tqdm pyyaml ipdb"
conda install scipy scikit-learn pandas matplotlib numpy tqdm pyyaml
echo "Install Open3d"
conda install open3d -c open3d-admin
