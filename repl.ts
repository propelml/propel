#!/usr/bin/env ts-node

import * as repl from "repl";
import * as propel from "./api";

// Wait for 1ms to allow node and tensorflow to print their junk.
setTimeout(() => {
  const context = repl.start("> ").context;
  context.$ = propel.$;
  context.propel = propel;
}, 1);
