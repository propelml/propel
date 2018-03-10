#!/usr/bin/env node
const run = require("./run");

process.on("unhandledRejection", e => { throw e; });

(async() => {
  run.sh("node tools/build.js");

  // Node.js tests
  run.tsnode("tools/test_node.ts", {"PROPEL": "dl"});   // DL backend
  run.tsnode("tools/test_node.ts", {"PROPEL": "tf"});   // TF backend
  // This one doesn't need to be run for different backends.
  run.tsnode("./tools/gendoc_test.ts");

  // Web browser tests
  run.tsnode("tools/test_browser.ts use-render");

  run.tsnode("tools/jasmine_shim_test.ts");
})();
