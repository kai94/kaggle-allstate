#!/bin/bash

sudo apt-get update
sudo apt-get install python-numpy python-pandas emacs24 python-pip awscli -y

export LC_ALL=C
pip install sklearn xgboost
