#!/bin/bash

COMMON="--mesh data/banana/banana.stl --num-points 10000 --radius 0.5 --width 640 --height 480 --fov 60"

echo "== Family 1R: el = 0° =="
python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/banana/n08_el0 \
  --num-templates 8 \
  --elev-list "0" \
  $COMMON

python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/banana/n12_el0 \
  --num-templates 12 \
  --elev-list "0" \
  $COMMON


echo "== Family 2R: el = -30°, +30° (N = 10, 20, 30) =="
python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/banana/n10_elpm30 \
  --num-templates 10 \
  --elev-list "-30, 30" \
  $COMMON

python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/banana/n20_elpm30 \
  --num-templates 20 \
  --elev-list "-30, 30" \
  $COMMON

python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/banana/n30_elpm30 \
  --num-templates 30 \
  --elev-list "-30, 30" \
  $COMMON


echo "== Family 3R: el = -30°, 0°, +30° (N = 12, 18, 24, 30) =="
python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/banana/n12_el0_pm30 \
  --num-templates 12 \
  --elev-list "-30, 0, 30" \
  $COMMON

python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/banana/n18_el0_pm30 \
  --num-templates 18 \
  --elev-list "-30, 0, 30" \
  $COMMON

python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/banana/n24_el0_pm30 \
  --num-templates 24 \
  --elev-list "-30, 0, 30" \
  $COMMON

python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/banana/n30_el0_pm30 \
  --num-templates 30 \
  --elev-list "-30, 0, 30" \
  $COMMON


echo "== Family 4R: el = -45°, -15°, +15°, +45° (N = 12, 16, 20, 24) =="
python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/banana/n12_elpm15_pm45 \
  --num-templates 12 \
  --elev-list "-45, -15, 15, 45" \
  $COMMON

python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/banana/n16_elpm15_pm45 \
  --num-templates 16 \
  --elev-list "-45, -15, 15, 45" \
  $COMMON

python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/banana/n20_elpm15_pm45 \
  --num-templates 20 \
  --elev-list "-45, -15, 15, 45" \
  $COMMON

python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/banana/n24_elpm15_pm45 \
  --num-templates 24 \
  --elev-list "-45, -15, 15, 45" \
  $COMMON


echo "== Family 5R: el = -60°, -30°, 0°, +30°, +60° (N = 10, 15, 20, 25, 30) =="
python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/banana/n10_el0_pm30_pm60 \
  --num-templates 10 \
  --elev-list "-60, -30, 0, 30, 60" \
  $COMMON

python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/banana/n15_el0_pm30_pm60 \
  --num-templates 15 \
  --elev-list "-60, -30, 0, 30, 60" \
  $COMMON

python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/banana/n20_el0_pm30_pm60 \
  --num-templates 20 \
  --elev-list "-60, -30, 0, 30, 60" \
  $COMMON

python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/banana/n25_el0_pm30_pm60 \
  --num-templates 25 \
  --elev-list "-60, -30, 0, 30, 60" \
  $COMMON

python scripts/generate_visible_templates.py \
  --outdir data/template_experiment/banana/n30_el0_pm30_pm60 \
  --num-templates 30 \
  --elev-list "-60, -30, 0, 30, 60" \
  $COMMON
