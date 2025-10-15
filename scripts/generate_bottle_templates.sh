#!/bin/bash

COMMON="--mesh data/bottle/bottle.stl --num-points 10000 --radius 0.5 --width 640 --height 480 --fov 60"

# n=5
python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/bottle/n25_el0_pm30_pm60 \
  --num-templates 25 \
  --elev-list "-60, -30, 0, 30, 60" \
  $COMMON
