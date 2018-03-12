#!/usr/bin/env node
const run = require("./run");
const { execSync } = require("child_process");
run.sh("./tools/website_render.js prod gendoc");
// pip install awscli
execSync(
  "aws s3 sync build/website_render/ s3://propelml.org --follow-symlinks --delete",
  {
    stdio: "inherit"
  }
);
