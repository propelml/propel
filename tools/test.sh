#!/bin/bash
set -e
cd `dirname "$0"`; cd ..
./node_modules/typescript/bin/tsc

# TODO TensorFlow backend tests.
node dist/binding_test.js
#node dist/tensor_test.js
#node dist/util_test.js
#node dist/binding_test.js

# Web backend tests.
export PROPEL="web"
node dist/backprop_test.js
node dist/tensor_test.js
node dist/util_test.js
