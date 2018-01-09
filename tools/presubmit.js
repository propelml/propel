#!/usr/bin/env node
const run = require("./run");
run.rmrf("./build");
run.sh("node ./tools/cpplint.js");
run.sh("node ./tools/tslint.js");
run.sh("node ./tools/stylelint.js");
run.sh("node ./tools/test.js")
