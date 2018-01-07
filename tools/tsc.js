#!/usr/bin/env node
const run = require('./run');
const extra = process.argv.slice(2).join(" ");
run.sh(`node ${run.root}/node_modules/typescript/bin/tsc
  ${extra} -p ${run.root}/tsconfig.json`)
