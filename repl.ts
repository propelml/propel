#!/usr/bin/env ts-node
/*
   Copyright 2018 Propel http://propel.site/.  All rights reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

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
