#!/usr/bin/env node
const run = require("./run");
run.sh(`node ./node_modules/http-server/bin/http-server --cors build/dev_website`);
