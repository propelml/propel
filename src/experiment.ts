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
  checkpointInterval?: number; // In min.
  checkpointsToKeep?: number;
}

const defaultOpts: ExperimentOpts = {
  checkpointInterval: 0.5,
  checkpointsToKeep: 3,
};

export async function experiment(name: string,
                                 opts?: ExperimentOpts): Promise<Experiment> {
  let exp: Experiment;
  exp = new DiskExperiment(name, opts);
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

abstract class Experiment {
  protected currentParams: Params;
  protected step_?: number;
  protected opts: ExperimentOpts;
  protected lastSave?: Date;

  constructor(public name: string, opts: ExperimentOpts) {
    this.opts = Object.assign(defaultOpts, opts);
  }

  abstract createOrRestore(): Promise<void>;

  get step() {
    return this.step_;
  }

  // Creates a checkpoint.
  abstract async save(): Promise<void>;

  // Gets a list of checkpoints. Most recent first.
  abstract async checkpoints(): Promise<number[]>;

  abstract async deleteCheckpoint(step: number): Promise<void>;

  private async maybeSave(): Promise<void> {
    if (this.lastSave) {
      const minSinceLastSave =
        ((new Date()).valueOf() - this.lastSave.valueOf()) / 60000;
      if (minSinceLastSave < this.opts.checkpointInterval) {
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
    this.maybeSave();
  }

  /** Performs SGD given the loss and current parameters. */
  sgd(opts: SGDOpts, lossFn: LossFn) {
    return this.minimize(sgd, opts, lossFn);
  }
}

///////////////////////

// In Node we have a DiskExperiment, which stores experiments to disk.
import * as fs from "fs";
import * as path from "path";
import * as rimraf from "rimraf";
import * as npy from "./npy";

function isDir(p: string): boolean {
  try {
    return fs.lstatSync(p).isDirectory();
  } catch (e) {
    if (e.code === "ENOENT") return false;
    throw e;
  }
}

function filePatternSearch(p: string, pattern: RegExp): string[] {
  if (isDir(p)) {
    let results = [];
    for (const fn of fs.readdirSync(p)) {
      const fullPath = path.join(p, fn);
      results = results.concat(filePatternSearch(fullPath, pattern));
    }
    return results;
  } else {
    if (path.basename(p).match(pattern)) {
      return [p];
    } else {
      return [];
    }
  }
}

function ensureDirExists(p: string): void {
  if (!isDir(p)) {
    ensureDirExists(path.dirname(p));
    fs.mkdirSync(p);
  }
}

class DiskExperiment extends Experiment {
  get dir() {
    const propelDir = path.join(process.env.HOME, ".propel/");
    return path.join(propelDir, this.name);
  }

  async createOrRestore(): Promise<void> {
    ensureDirExists(this.dir);
    const checkpoints = await this.checkpoints();
    console.log("checkpoints", checkpoints);
    if (checkpoints.length > 0) {
      const latestCheckpoint = checkpoints[0];
      console.log("Restore checkpoint", latestCheckpoint);
      await this.restore(latestCheckpoint);
      assert(this.step_ === latestCheckpoint);
    } else {
      this.currentParams = createParams();
      this.step_ = 0;
    }
  }

  async restore(step: number): Promise<Params> {
    const p = this.checkpointPath(step);
    const npyFiles = filePatternSearch(p, /\.npy$/);
    const params = createParams();
    for (const fn of npyFiles) {
      const name = fn.replace(p, "").replace(/\.npy$/, "");
      const tensor = await npy.load(fn);
      params.set(name, tensor);
      console.log(name, tensor.dtype, tensor.shape);
    }
    this.step_ = step;
    this.currentParams = params;
    return params;
  }

  async checkpoints(): Promise<number[]> {
    const out = [];
    for (const fn of fs.readdirSync(this.dir)) {
      if (fn.match(/^\d+$/)) {
        out.push(Number(fn));
      }
    }
    return out.sort((a, b) => b - a);
  }

  async deleteCheckpoint(step: number): Promise<void> {
    const p = this.checkpointPath(step);
    rimraf.sync(p);
    await print("Checkpoint deleted:", p);
  }

  private checkpointPath(step: number): string {
    return path.join(this.dir, String(step).padStart(8, "0")) + "/";
  }

  async save(): Promise<void> {
    const checkpointPath = this.checkpointPath(this.step);
    let totalSize = 0;
    for (const [name, tensor] of this.currentParams) {
      const tensorPath = path.join(checkpointPath, name) + ".npy";
      ensureDirExists(path.dirname(tensorPath));
      const ab = await npy.serialize(tensor);
      totalSize += ab.byteLength;
      fs.writeFileSync(tensorPath, new Buffer(ab));
    }
    const megs = (totalSize / (1024 * 1024)).toFixed(2);
    await print(`Checkpoint saved. Size: ${megs}M.`, checkpointPath);
  }
}
