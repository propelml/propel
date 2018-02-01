#!/usr/bin/env node
// This file is just to easily run typescript from the command-line without
// requiring that ts-node is installed. All of the logic for building the
// website is in tools/build_website_impl.ts
require("ts-node").register({"typeCheck": true });
require("./build_website_impl.ts");
