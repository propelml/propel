#!/usr/bin/env node
const { execSync } = require("child_process");
// pip install awscli
execSync("aws s3 sync website/ s3://propelml.org --follow-symlinks --delete", {
  stdio: "inherit"
});
