#!/usr/bin/env node
const run = require('./run');
let extra = process.argv.slice(2).join(" ");
if (!process.env.APPVEYOR) extra += " --progress";
run.sh(`node ./node_modules/webpack/bin/webpack.js
  --config webpack.config.js
  ${extra}
`);
