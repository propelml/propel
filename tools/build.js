#!/usr/bin/env node
const run = require("./run");

if (process.argv.indexOf("clean") >= 0) {
  run.rmrf("./build");
}

(async() => {
  run.sh("node ./tools/build_binding.js");
  run.sh("node ./tools/build_website.js");
  await run.parcel("src/test_dl.ts", "build/website", true);
  await run.parcel("website/test_website.ts", "build/website", true);
})();
