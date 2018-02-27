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

interface ExperimentOpts {
  printStepSecs: number;
}

const expOptDefaults: ExperimentOpts = {
  printStepSecs: 1,
};

export async function experiment(name: string,
    opts?: ExperimentOpts): Promise<Experiment> {
  const exp = new Experiment(name, opts);
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

interface RateInfo {
  step: number;
  time: Date;
}

class Experiment {
  private currentParams: Params;
  private step_?: number;
  readonly opts: ExperimentOpts;
  private lastPrint?: Date;
  private rateHistory: RateInfo[] = [];

  constructor(readonly name: string, opts: ExperimentOpts) {
    this.opts = Object.assign(expOptDefaults, opts);
  }

  /** Loads from disk */
  async createOrRestore(): Promise<void> {
    // Faking it now.
    this.currentParams = createParams();
    this.step_ = 0;
  }

  get step() {
    return this.step_;
  }

  /** Performs SGD given the loss and current parameters. */
  sgd(opts: SGDOpts, lossFn: LossFn) {
    return this.minimize(sgd, opts, lossFn);
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

    const printArgs = ["step", this.step, "loss", loss];
    const rate = this.getRate();
    if (rate) {
      printArgs.push("steps/sec");
      printArgs.push(rate.toFixed(1));
    }

    this.printProgress(...printArgs).then(() => loss.dispose());
  }

  private getRate(): number {
    this.updateRateHistory();
    const first = this.rateHistory[0];
    const last = this.rateHistory[this.rateHistory.length - 1];
    const timeDiff = (last.time.valueOf() - first.time.valueOf()) / 1000;
    const stepDiff = last.step - first.step;
    return stepDiff / timeDiff;
  }

  private updateRateHistory() {
    const last = this.rateHistory[this.rateHistory.length - 1];
    if (!last || (secsSince(last.time) > 0.5 && this.step !== last.step)) {
      this.rateHistory.push({
        step: this.step,
        time: new Date(),
      });
      // We don't need too many elements.
      if (this.rateHistory.length > 3) {
        this.rateHistory.shift();
      }
    }
  }

  private async printProgress(...args: PrintArgs): Promise<void> {
    // Drop print if it's been less than 1 second (or whatever
    // this.opts.printStepSecs is set to).
    if (this.lastPrint) {
      if (secsSince(this.lastPrint) < this.opts.printStepSecs) {
        return;
      }
    }
    this.lastPrint = new Date();
    await print(...args);
  }
}

function secsSince(t: Date): number {
  return (new Date().valueOf() - t.valueOf()) / 1000;
}
