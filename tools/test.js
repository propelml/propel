#!/usr/bin/env node
const run = require("./run");

// Build the project.
run.sh('node ./tools/build.js')
run.sh('node ./tools/tsc.js')
run.sh('node ./tools/webpack.js')
// Node.js tests
run.sh("node dist/test_node.js", {"PROPEL": "dl"})     // DL backend
run.sh("node dist/test_node.js", {"PROPEL": "tf"})     // TF backend
run.sh("node dist/binding_test.js", {"PROPEL": "tf"})  // TF-only test. 
// Web browser tests
run.sh('node dist/test_browser.js');
