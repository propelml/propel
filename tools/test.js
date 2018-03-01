#!/usr/bin/env node
const run = require("./run");

process.on("unhandledRejection", e => { throw e; });

(async() => {
  run.sh("node tools/build.js");

  // Node.js tests
  run.tsnode("src/test_node.ts", {"PROPEL": "dl"});     // DL backend
  run.tsnode("src/test_node.ts", {"PROPEL": "tf"});     // TF backend
  run.tsnode("src/binding_test.ts", {"PROPEL": "tf"});  // TF-only test.

  // Web browser tests
  run.tsnode("tools/test_browser.ts");

  run.tsnode("tools/jasmine_shim_test.ts");
})();
