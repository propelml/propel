#!/bin/bash
set -e -v
cd `dirname "$0"`; cd ..
# GYP normally hides the exact commands being run unless V=1 is set.
export V=1
./tools/tslint.sh
./deps/cpplint/cpplint.py *.cc
npm build .
node ./node_modules/webpack/bin/webpack.js
bash ./tools/test.sh
# PASS
