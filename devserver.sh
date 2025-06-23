#!/bin/sh
source .venv/bin/activate
python -u -m flask --app /home/user/end-to-end-ml/application.py run --debug