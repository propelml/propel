#!/usr/bin/env node
const run = require("./run");
run.sh("python ./deps/cpplint/cpplint.py src/binding.cc src/check.h");
