#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Legacy entry point. Use fairseq_cli/train.py or fairseq-train instead.
"""

from fairseq_cli.train import cli_main

# libgcc_s.so.1 must be installed for pthread_cancel to work
import ctypes

if __name__ == "__main__": 
    #libgcc_s = ctypes.CDLL('libgcc_s.so.1')
    cli_main()
