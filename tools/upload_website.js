#!/usr/bin/env node
const run = require("./run");
const { execSync } = require("child_process");
run.sh("node ./tools/build_website.js");
// pip install awscli
execSync("aws s3 sync build/website/ s3://propelml.org --follow-symlinks --delete", {
  stdio: "inherit"
});
