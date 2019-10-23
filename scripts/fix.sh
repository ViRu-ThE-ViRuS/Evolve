#!/bin/sh
echo "Running fixer..."

echo ">>autopep8"
autopep8 --global-config .pylint --in-place --aggressive --aggressive --aggressive **/*.py *.py
