#!/bin/bash
cd /Users/hanna/Downloads/GLASSBOX
lldb -b -o "run" -o "bt" -o "quit" build/train_scale -- --rung6 --resume build/model_rung6.bin --steps 600 --lr 0.0008
