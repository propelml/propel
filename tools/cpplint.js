#!/usr/bin/env node
const run = require("./run");
run.sh("python ./deps/cpplint/cpplint.py src/tf_binding.cc src/check.h");
