#!/usr/bin/env node
// To just run the linter: ./tools/tslint.js
// To auto-format code:    ./tools/tslint.js --fix
const run = require("./run");
const extra = process.argv.slice(2).join(" ");
run.sh(`node ./node_modules/tslint/bin/tslint
  ${extra} -p ${run.root}`);
