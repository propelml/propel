#!/usr/bin/env node
const run = require("./run");

// Build the project.
run.sh("node ./tools/build_binding.js");
run.sh("node ./tools/build_website.js");
run.parcel("test_dl.ts", "build/website", true);
run.parcel("test_isomorphic.ts", "build/website", true);

// Node.js tests
run.tsnode("test_node.ts", {"PROPEL": "dl"});     // DL backend
run.tsnode("test_node.ts", {"PROPEL": "tf"});     // TF backend
run.tsnode("binding_test.ts", {"PROPEL": "tf"});  // TF-only test.

// Web browser tests
run.tsnode("test_browser.ts");
