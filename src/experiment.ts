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

// Experiment abstracts the concept of saved parameters.
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
import { assert, getOutputHandler, IS_NODE } from "./util";

export interface ExperimentOpts {
  checkpointsToKeep?: number;
  printStepSecs?: number;
  saveOnExit?: boolean;
  saveSecs?: number;
}

const defaultOpts: ExperimentOpts = {
  checkpointsToKeep: 3,
  printStepSecs: 1,
  saveOnExit: true,
  saveSecs: 60,
};

export async function experiment(name: string,
                                 opts?: ExperimentOpts): Promise<Experiment> {
  let exp: Experiment;
  if (name === "cache") {
    throw Error("Invalid experiment name.");
  }
  if (IS_NODE) {
    const { DiskExperiment } = require("./disk_experiment");
    exp = new DiskExperiment(name, opts);
  } else {
    exp = new BrowserExperiment(name, opts);
  }
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
    // TODO this is a hack to get cell output working on the homepage. A better
    // solution would be to globally replace the console object in sandbox.ts.
    const oh = getOutputHandler();
    const msg = await printHelper(...args);
    if (oh) {
      oh.print(msg);
    } else {
      console.log(msg);
    }
  });
  await lastPrint;
}

async function printHelper(...args: PrintArgs): Promise<string> {
  const strings = await Promise.all(args.map(async(arg) => {
    if (arg instanceof Tensor) {
      return format.toString(arg);
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

export abstract class Experiment {
  private lastPrint?: Date;
  private rateHistory: RateInfo[] = [];
  protected currentParams: Params;
  protected lastSave?: Date;
  protected step_?: number;
  readonly opts: ExperimentOpts;

  constructor(readonly name: string, opts?: ExperimentOpts) {
    this.opts = Object.assign(defaultOpts, opts);
  }

  get step() {
    return this.step_;
  }

  get params() {
    return this.currentParams;
  }

  /** Performs SGD given the loss and current parameters. */
  sgd(opts: SGDOpts, lossFn: LossFn): void {
    return this.minimize(sgd, opts, lossFn);
  }

  abstract createOrRestore(): Promise<void>;

  /** Saves a new checkpoint. */
  abstract async save(): Promise<void>;

  /** Gets a list of checkpoints. Most recent first. */
  abstract async checkpoints(): Promise<number[]>;

  abstract async deleteCheckpoint(step: number): Promise<void>;

  private async maybeSave(): Promise<void> {
    if (this.lastSave) {
      if (secsSince(this.lastSave) < this.opts.saveSecs) {
        // Bail out if we've saved less than saveSecs seconds ago.
        return;
      }
    }
    await this.save();
    this.lastSave = new Date();

    // Delete old checkpoints.
    const checkpoints = await this.checkpoints();
    for (let i = 0; i < checkpoints.length; i++) {
      if (i >= this.opts.checkpointsToKeep) {
        this.deleteCheckpoint(checkpoints[i]);
      }
    }
  }

  /** Modifies the current parameters given the loss and optimizer. */
  minimize(optimizer: Optimizer, opts: SGDOpts, lossFn: LossFn): void {
    this.step_++;
    let loss;
    gc((keep) => {
      const gradFn = gradParams(lossFn);
      // Forward/Backward pass
      this.currentParams.isTraining = true;
      const gradsAndLoss = gradFn(this.currentParams);
      this.currentParams.isTraining = false;
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
    this.maybeSave();
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

// The browser doesn't save checkpoints at all right now.
class BrowserExperiment extends Experiment {
  async createOrRestore(): Promise<void> {
    this.currentParams = createParams();
    this.step_ = 0;
  }

  async save(): Promise<void> { }

  async checkpoints(): Promise<number[]> {
    return [];
  }

  async deleteCheckpoint(step: number): Promise<void> { }
}
