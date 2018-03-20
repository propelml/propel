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
import { Transpiler } from "./nb_transpiler";
import { RPC, WindowRPC } from "./rpc";

async function importModule(target) {
  const m = {
    matplotlib,
    mnist,
    propel,
  }[target];
  if (m) {
    return m;
  }
  throw new Error("Unknown module: " + target);
}

let lastExecutedCellId: number = null;

const transpiler = new Transpiler();

const channelId =
    document.querySelector("meta[name=rpc-channel-id]").getAttribute("content");
const rpc: RPC = new WindowRPC(window.parent, channelId);
rpc.start({ runCell });

async function runCell(source: string, cellId: number): Promise<void> {
  lastExecutedCellId = cellId;
  try {
    const console = new Console(rpc, cellId);
    const transpiledSource = transpiler.transpile(source, `cell${cellId}`);
    const fn = globalEval(transpiledSource);
    const result = await fn(global, importModule, console);
    if (result !== undefined) {
      console.log(result);
    }
  } catch (e) {
    const message = transpiler.formatException(e);
    rpc.call("print", cellId, message);
    // When running tests, rethrow any errors. This ensures that errors
    // occurring during notebook cell evaluation result in test failure.
    if (window.navigator.webdriver) {
      throw e;
    }
  }
}

function guessCellId(error?: Error): number {
  const name = transpiler.getEntryPoint(error);
  if (name != null) {
    const m = name.match(/cell(\d+)/);
    if (m) return +m[1];
  }
  return lastExecutedCellId;
}

class Console {
  constructor(private rpc: RPC, private cellId: number) { }

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

  private print(...args: any[]) {
    this.rpc.call("print", this.cellId, args.map(this.inspect).join(" "));
  }

  log(...args: any[]): void {
    this.print(...args);
  }

  warn(...args: any[]): void {
    this.print(...args);
  }

  error(...args: any[]): void {
    this.print(...args);
  }
}

matplotlib.setOutputHandler({
  imshow(data: any): void {
    rpc.call("imshow", guessCellId(), data);
  },

  plot(data: any): void {
    rpc.call("plot", guessCellId(), data);
  },

  print(data: any): void {
    rpc.call("print", guessCellId(), data);
  }
});

window.addEventListener("error", (ev: ErrorEvent) => {
  let cellId, message;
  if (ev.error != null) {
    cellId = guessCellId(ev.error);
    message = transpiler.formatException(ev.error);
  } else {
    cellId = guessCellId();
    message = ev.message;
  }
  rpc.call("print", cellId, message);
});

// TODO: also handle unhandledrejection. This should work in theory, in Chrome
// at least; however I was unable to trigger this event.
