#!/usr/bin/env bash

set -e

PYTHON=python

$PYTHON -m src.main_infer \
  --pdf data/input_pdfs \
  --config config/config.yaml \
  --output_json data/outputs/json/predictions_example.json
