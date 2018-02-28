#!/usr/bin/env node
const run = require("./run");
run.sh(`node ./node_modules/http-server/bin/http-server build/website`);
