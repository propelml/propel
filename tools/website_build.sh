#!/bin/sh
set -e -v
cd `dirname "$0"`; cd ..
B=./node_modules/markdown-to-html/bin/markdown
$B README.md -s style.css -c propelml/propel > website/index.html
./tools/webpack
# Now call ./tools/website_upload.sh
