#!/bin/bash
set -e -v
cd `dirname "$0"`; cd ..
./node_modules/typescript/bin/tsc

# Backend independent
node dist/gendoc_test.js

# TensorFlow backend
node dist/binding_test.js  # TF only

node dist/api_test.js
node dist/backend_test.js
node dist/utils_test.js
node dist/mnist_test.js
node dist/nb_transpiler_test.js
node dist/format_test.js
node dist/nn_example_test.js

# Now using the Deep Learn backend.
export PROPEL="dl"
node dist/api_test.js
node dist/backend_test.js
node dist/utils_test.js
node dist/mnist_test.js
node dist/nb_transpiler_test.js
node dist/format_test.js
node dist/nn_example_test.js
