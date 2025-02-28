#!/bin/bash

# python train.py --model linear_signorm --x0_dataset moons --x1_dataset squares
# python train.py --model vp_signorm --x0_dataset moons --x1_dataset squares
# python train.py --model linear_signorm --x0_dataset moons --x1_dataset normal
# python train.py --model linear_signorm --x0_dataset squares --x1_dataset normal
# python train.py --model linear_signorm --x0_dataset deltas --x1_dataset normal
# python train.py --model vp_signorm --x0_dataset moons --x1_dataset normal
# python train.py --model vp_signorm --x0_dataset squares --x1_dataset normal
# python train.py --model vp_signorm --x0_dataset deltas --x1_dataset normal
python train.py --model flow --x0_dataset moons
python train.py --model flow --x0_dataset squares
