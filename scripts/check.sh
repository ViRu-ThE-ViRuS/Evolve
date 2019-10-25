#!/bin/sh
echo "Running checks..."

echo ">>pylint"
pylint **/**.py *.py

echo ">>pytest"
pytest

echo ">>cleanup"
./scripts/cleanup.sh