#!/bin/bash
cd `dirname "$0"`; cd ..
set -e
./tools/webpack
ts-node backprop_test.ts
ts-node tensor_test.ts
ts-node util_test.ts
echo PASS
