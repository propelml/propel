/*!
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

import * as propel from "../src/api";
import * as matplotlib from "../src/matplotlib";
import * as mnist from "../src/mnist";

import { global, globalEval } from "../src/util";
import { transpile } from "./nb_transpiler";
import { SandboxRPC } from "./sandbox_rpc";

async function importModule(target) {
  const m = {
    matplotlib,
    mnist,
    propel
  }[target];
  if (m) {
    return m;
  }
  throw new Error("Unknown module: " + target);
}

let lastExecutedCellId: number = null;

const rpc = new SandboxRPC(window.parent, {
  async runCell(source: string, cellId: number): Promise<void> {
    lastExecutedCellId = cellId;
    try {
      const console = new Console(rpc, cellId);
      source = transpile(source);
      source += `\n//# sourceURL=__cell${cellId}__.js`;
      const fn = globalEval(source);
      const result = await fn(global, importModule, console);
      if (result !== undefined) {
        console.log(result);
      }
    } catch (e) {
      const msg = e instanceof Error ? e.stack : `exception: ${e}`;
      rpc.call("console", cellId, msg);
      // When running tests, rethrow any errors. This ensures that errors
      // occurring during notebook cell evaluation result in test failure.
      if (window.navigator.webdriver) {
        throw e;
      }
    }
  }
});

function guessCellId(): number {
  const stacktrace = new Error().stack.split("\n");
  for (let i = stacktrace.length - 1; i >= 0; --i) {
    const line = stacktrace[i];
    const m = line.match(/__cell(\d+)__/);
    if (m) return +m[1];
  }
  return lastExecutedCellId;
}

class Console {
  constructor(private rpc: SandboxRPC, private cellId: number) {}

  private inspect(value): string {
    if (value instanceof propel.Tensor) {
      return value.toString();
    }
    if (value && typeof value === "object") {
      try {
        return JSON.stringify(value, null, 2);
      } catch (e) {}
    }
    return value + ""; // Convert to string.
  }

  private send(...args) {
    return this.rpc.call("console", this.cellId, ...args.map(this.inspect));
  }

  log(...args) {
    return this.send(...args);
  }

  warn(...args) {
    return this.send(...args);
  }

  error(...args) {
    return this.send(...args);
  }
}

matplotlib.setOutputHandler({
  plot(data: any): void {
    rpc.call("plot", guessCellId(), data);
  },

  imshow(data: any): void {
    rpc.call("imshow", guessCellId(), data);
  }
});
