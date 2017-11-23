#!/bin/bash
set -e -v
cd `dirname "$0"`; cd ..
./deps/cpplint/cpplint.py *.cc
npm build .
node ./node_modules/webpack/bin/webpack.js
./node_modules/typescript/bin/tsc
node dist/backprop_test.js
node dist/tensor_test.js
node dist/util_test.js
node binding_test.js
echo PASS
