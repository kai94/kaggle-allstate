#!/bin/bash

sudo apt-get update
sudo apt-get install python-numpy python-pandas emacs24 pip-y

export LC_ALL=C
pip install sklearn xgboost
