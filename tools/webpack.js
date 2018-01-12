#!/usr/bin/env node
const run = require('./run');
const extra = process.argv.slice(2).join(" ");
run.sh(`node ./node_modules/webpack/bin/webpack.js
  --config webpack.config.js
  --progress
  ${extra}
`);
