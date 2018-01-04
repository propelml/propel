#!/usr/bin/env node
const run = require('./run');
run.sh(`ts-node gendoc.ts`);
// pip install aws
run.sh(`aws s3 sync website/ s3://propelml.org --follow-symlinks --delete`);
