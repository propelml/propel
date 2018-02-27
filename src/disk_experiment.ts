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

// In Node we have a DiskExperiment, which stores experiments to disk.
// This is included in a separate file so it doesn't get included in
// the browser bundle.

import * as fs from "fs";
import * as path from "path";
import * as rimraf from "rimraf";
import { Experiment, print }  from "./experiment";
import * as npy from "./npy";
import { Params, params as createParams } from "./params";
import { assert } from "./util";

function propelDir(): string {
  if (process.env.PROPEL_DIR) {
    return process.env.PROPEL_DIR;
  } else {
    return path.join(process.env.HOME, ".propel/");
  }
}

export class DiskExperiment extends Experiment {
  get dir() {
    return path.join(propelDir(), this.name);
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

export function isDir(p: string): boolean {
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
