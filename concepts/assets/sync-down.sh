#! /bin/bash
#
# sync-down.sh
# Copyright (C) 2023 Jiayuan Mao <maojiayuan@gmail.com>
#
# Distributed under terms of the MIT license.
#

set -e

wget -O assets.tar https://concepts-ai.com/assets.tar

# extract the assets and do overwrite
tar -xvf assets.tar

# remove the tar file
rm assets.tar
