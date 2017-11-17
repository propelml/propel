#!/bin/bash
cd `dirname "$0"`
# pip install aws
aws s3 sync ../website/ s3://propelml.org --follow-symlinks --delete
