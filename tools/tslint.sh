#!/bin/bash
# To just run the linter: ./tools/tslint.sh
# To auto-format code:    ./tools/tslint.sh --fix
cd `dirname "$0"`; cd ..
exec ./node_modules/tslint/bin/tslint --project . -e "**/deeplearnjs/**/*" $@
