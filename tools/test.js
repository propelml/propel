#!/usr/bin/env node
const run = require("./run");

// Build the project.
run.sh("node ./tools/build_binding.js");
run.sh("node ./tools/tsc.js");
run.sh("node ./tools/build_website.js");
run.parcel("test_dl.ts", "build/website", true);
run.parcel("test_isomorphic.ts", "build/website", true);

// Node.js tests
run.sh("node build/test_node.js", {"PROPEL": "dl"});     // DL backend
run.sh("node build/test_node.js", {"PROPEL": "tf"});     // TF backend
run.sh("node build/binding_test.js", {"PROPEL": "tf"});  // TF-only test. 

// Web browser tests
run.sh("node build/test_browser.js");
