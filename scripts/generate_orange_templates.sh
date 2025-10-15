#!/bin/bash

COMMON="--mesh data/orange/orange.stl --num-points 10000 --radius 0.5 --width 640 --height 480 --fov 60"

# n=5
python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/orange/n05 \
  --num-templates 5 \
  --elev-list "-85.0, -29.9, 0.0, 29.9, 85.0" \
  $COMMON
