#!/usr/bin/env node
// To just run the linter: ./tools/stylelint.js
// To auto-format code:    ./tools/stylelint.js --fix
const run = require('./run');
const extra = process.argv.slice(2).join(" ");
// TODO use *.scss here.
run.sh(`node ./node_modules/stylelint/bin/stylelint.js
  ${extra}
  --config stylelint.json
  website/main.scss
  website/normalize.scss
  website/skeleton.scss
  website/syntax.scss
  website/syntax_dark.scss
  website/syntax_light.scss
`);
