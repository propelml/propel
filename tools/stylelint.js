#!/usr/bin/env node
// To just run the linter: ./tools/stylelint.js
// To auto-format code:    ./tools/stylelint.js --fix
const run = require('./run');
const extra = process.argv.slice(2).join(" ");
run.sh(`node ./node_modules/stylelint/bin/stylelint.js
  ${extra}
  --config stylelint.json
  website/style.css
  website/syntax.css`);
