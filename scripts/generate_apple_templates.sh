#!/bin/bash

COMMON="--mesh data/apple/apple.stl --num-points 10000 --radius 0.5 --width 640 --height 480 --fov 60"

# n=1
python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/apple/n01 \
  --num-templates 1 \
  --elev-list "0.0" \
  $COMMON

# n=2
python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/apple/n02 \
  --num-templates 2 \
  --elev-list "-85.0, 85.0" \
  $COMMON

# n=3
python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/apple/n03 \
  --num-templates 3 \
  --elev-list "-85.0, 0.0, 85.0" \
  $COMMON

# n=4
python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/apple/n04 \
  --num-templates 4 \
  --elev-list "-85.0, -19.4, 19.4, 85.0" \
  $COMMON

# n=5
python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/apple/n05 \
  --num-templates 5 \
  --elev-list "-85.0, -29.9, 0.0, 29.9, 85.0" \
  $COMMON

# n=6
python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/apple/n06 \
  --num-templates 6 \
  --elev-list "-85.0, -36.7, -11.5, 11.5, 36.7, 85.0" \
  $COMMON

# n=7
python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/apple/n07 \
  --num-templates 7 \
  --elev-list "-85.0, -41.6, -19.4, 0.0, 19.4, 41.6, 85.0" \
  $COMMON

# n=8
python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/apple/n08 \
  --num-templates 8 \
  --elev-list "-85.0, -45.4, -25.3, -8.2, 8.2, 25.3, 45.4, 85.0" \
  $COMMON

# n=9
python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/apple/n09 \
  --num-templates 9 \
  --elev-list "-85.0, -48.3, -29.9, -14.4, 0.0, 14.4, 29.9, 48.3, 85.0" \
  $COMMON

# n=10
python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/apple/n10 \
  --num-templates 10 \
  --elev-list "-85.0, -50.8, -33.6, -19.4, -6.4, 6.4, 19.4, 33.6, 50.8, 85.0" \
  $COMMON
