#!/usr/bin/env node
const run = require('./run');
// pip install aws
run.sh(`aws s3 sync website/ s3://propelml.org --follow-symlinks --delete`);
