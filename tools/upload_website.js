#!/usr/bin/env node
const run = require("./run");
const { execSync } = require("child_process");
run.sh("node ./node_modules/ts-node/dist/bin.js tools/build_website.ts");
// pip install awscli
execSync("aws s3 sync build/website/ s3://propelml.org --follow-symlinks --delete", {
  stdio: "inherit"
});
