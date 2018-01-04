#!/usr/bin/env node
const run = require("./run");
run.sh("python ./deps/cpplint/cpplint.py binding.cc check.h");
