#!/bin/sh
srun --pty -p gpu-long --gres=gpu --mem=100G /bin/bash

conda activate py39
