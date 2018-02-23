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

// Experiement abstracts the concept of saved parameters.
// The scope of this module include
//  - saving parameters to disk
//  - restoring parameters from disk
//  - providing interfaces to optimizers
//    To make sure that backprop tracing does not leak memory
//    loss must be calculated inside a function with the parameters
//    provided. Given that experiements handle parameters, this
//    provides a good place to interface.

import { gradParams } from "./backprop";
import * as format from "./format";
import { Params, params as createParams } from "./params";
import { gc, NamedTensors, Tensor } from "./tensor";
import { assert } from "./util";

export async function experiment(name: string, opts = {}): Promise<Experiment> {
  const exp = new Experiment(name);
  await exp.createOrRestore();
  return exp;
}

type PrintArgs = Array<number | string | Tensor>;
let lastPrint: Promise<void> = Promise.resolve();

/** A convenience function for printing tensors without calling dataSync on
 * them. The following code will not block the browser.
 *
 *    import * as pr from "propel"
 *    pr.print("hello", pr.range(3));
 */
export async function print(...args: PrintArgs): Promise<void> {
  // All this is to make sure that log items are in the order they came.
  lastPrint = lastPrint.then(async() => {
    console.log(await printHelper(...args));
  });
  await lastPrint;
}

async function printHelper(...args: PrintArgs): Promise<string> {
  const strings = await Promise.all(args.map(async(arg) => {
    if (arg instanceof Tensor) {
      return format.toString(arg.shape, await arg.data());
    } else {
      return arg;
    }
  }));
  return strings.join(" ");
}

export interface SGDOpts {
  lr: number;
  momentum?: number; // TODO currently unused.
}

type LossFn = (params: Params) => Tensor;

// Optimizer is expected to modify the params in someway.
type Optimizer = (opts, params: Params, grads: NamedTensors) => void;

function sgd(opts, params: Params, grads: NamedTensors): void {
  for (const name of Object.keys(grads)) {
    const g = grads[name];
    const p = params.get(name);
    // p -= g * lr
    p.assign(p.sub(g.mul(opts.lr)));
  }
}

class Experiment {
  private currentParams: Params;
  private step_?: number;

  constructor(public name: string) { }

  /** Loads from disk */
  async createOrRestore(): Promise<void> {
    // Faking it now.
    this.currentParams = createParams();
    this.step_ = 0;
  }

  get step() {
    return this.step_;
  }

  /** Modifies the current parameters given the loss and optimizer. */
  async minimize(optimizer: Optimizer, opts: SGDOpts, lossFn: LossFn) {
    this.step_++;
    let loss;
    gc((keep) => {
      const gradFn = gradParams(lossFn);
      // Forward/Backward pass
      const gradsAndLoss = gradFn(this.currentParams);
      const grads = gradsAndLoss[0];
      loss = gradsAndLoss[1];
      assert(loss.rank === 0);
      keep(loss);
      optimizer(opts, this.currentParams, grads);
    });

    print("step", this.step, "loss", loss).then(() => loss.dispose());
  }

  /** Performs SGD given the loss and current parameters. */
  sgd(opts: SGDOpts, lossFn: LossFn) {
    return this.minimize(sgd, opts, lossFn);
  }
}
