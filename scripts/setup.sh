#!/bin/sh
echo "Running setup..."

VENV_NAME="venv"

echo "install virtualenv"
pip3 install --user virtualenv

echo "setup virtualenv"
~/.local/bin/virtualenv --python=$(which python3) $VENV_NAME
. $VENV_NAME/bin/activate
pip3 install -r requirements.txt

echo "setup completed!"