#!/usr/bin/env node
// To just run the linter: ./tools/stylelint.js
// To auto-format code:    ./tools/stylelint.js --fix
const run = require('./run');
run.sh(`node ./node_modules/stylelint/bin/stylelint.js
  --config stylelint.json
  website/style.css
  website/syntax.css`);
