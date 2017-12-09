#!/bin/sh
set -e -v
cd `dirname "$0"`; cd ..
./tools/webpack
# Now call ./tools/website_upload.sh
