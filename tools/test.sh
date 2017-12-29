#!/bin/bash
set -e -v
cd `dirname "$0"`; cd ..

./node_modules/typescript/bin/tsc
node ./node_modules/webpack/bin/webpack.js

# Node.js tests
PROPEL=tf node dist/test_node.js    # TensorFlow backend
PROPEL=tf node dist/binding_test.js # TensorFlow-only test
PROPEL=dl node dist/test_node.js    # DeepLearn backend

# Web browser tests
node dist/test_browser.js
