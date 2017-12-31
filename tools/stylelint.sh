#!/bin/bash
# To just run the linter: ./tools/stylelint.sh
# To auto-format code:    ./tools/stylelint.sh --fix
cd `dirname "$0"`; cd ..
exec ./node_modules/.bin/stylelint \
  --config stylelint.json \
  website/style.css  \
  website/syntax.css  \
  $@
