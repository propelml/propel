#!/bin/bash
set -e -v
cd `dirname "$0"`; cd ..
./node_modules/typescript/bin/tsc

# TensorFlow backend
node dist/binding_test.js  # TF only

node dist/api_test.js
node dist/backend_test.js
node dist/util_test.js
node dist/mnist_test.js

# Now using the Deep Learn backend.
export PROPEL="web"
node dist/api_test.js
node dist/backend_test.js
node dist/util_test.js
node dist/mnist_test.js
