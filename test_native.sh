#!/bin/bash
set -e

echo "Running native video detector..."

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -r requirements.txt

mkdir -p input output logs trained_models

echo "Drop videos into ./input and monitor ./output"
python app.py
