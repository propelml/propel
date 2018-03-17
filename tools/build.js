#!/usr/bin/env node
const run = require("./run");

if (process.argv.indexOf("clean") >= 0) {
  run.rmrf("./build");
}

(async() => {
  run.sh("node ./tools/build_tf_binding.js");
  run.sh("node ./tools/website_render.js");
})();
