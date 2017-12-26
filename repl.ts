#!/usr/bin/env ts-node

import * as repl from "repl";
import { inspect } from "util";
import * as propel from "./api";

// This is currently node.js specific.
// TODO: move to api.js once it can be shared with the browser.
propel.Tensor.prototype[inspect.custom] = function(depth, opts) {
  return this.toString();
};

// Wait for 1ms to allow node and tensorflow to print their junk.
setTimeout(() => {
  const context = repl.start("> ").context;
  context.$ = propel.$;
  context.propel = propel;
}, 1);
