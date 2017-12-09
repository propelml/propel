#!/bin/bash
set -e -v
cd `dirname "$0"`; cd ..
./node_modules/.bin/typedoc \
  --exclude "**/{node_modules,deps}/**/*.ts"  \
  --out website/docs   \
  --entryPoint \"api\" \
  --excludeExternals \
  --readme none \
  --module commonjs \
  --theme minimal \
  .
