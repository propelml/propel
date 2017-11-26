#!/bin/bash
cd `dirname "$0"`; cd ..
exec ./node_modules/tslint/bin/tslint --project . -e "**/deeplearnjs/**/*"
