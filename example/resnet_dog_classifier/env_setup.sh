#!/usr/bin/env bash
rm -rf venv
python2 -m pip install virtualenv --user
python2 -m virtualenv venv
venv/bin/python -m pip install apache-beam[gcp]
