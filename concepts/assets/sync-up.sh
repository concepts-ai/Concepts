#! /bin/bash
#
# sync-up.sh
# Copyright (C) 2023 Jiayuan Mao <maojiayuan@gmail.com>
#
# Distributed under terms of the MIT license.
#

set -e

tar czvf assets.tar basic cliport objects robots visual_reasoning_datasets
#rsync -avP assets.tar droplet.jiayuanm.com:~/concepts_docs/html
rsync -avP assets.tar iris@droplet.jiayuanm.com:~/concepts_docs/html
rm assets.tar

