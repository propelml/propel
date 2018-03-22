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
import { Experiment, ExperimentOpts, print }  from "./experiment";
import * as npy from "./npy";
import { Params, params as createParams } from "./params";
import { assert } from "./util";
import { isDir, propelDir } from "./util_node";

export class DiskExperiment extends Experiment {
  constructor(readonly name: string, opts?: ExperimentOpts) {
    super(name, opts);
    if (this.opts.saveOnExit) {
      process.on("exit", (exitCode) => {
        // If there's no error, save the checkpoint one file last time.
        if (exitCode === 0 && this.step > 0) {
          this.save();
        }
      });
    }
  }

  get dir() {
    return path.join(propelDir(), this.name);
  }

  async createOrRestore(): Promise<void> {
    ensureDirExists(this.dir);
    const checkpoints = await this.checkpoints();
    if (checkpoints.length > 0) {
      console.log("checkpoints", checkpoints);
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
      let name = path.normalize(fn)
        .replace(p + path.sep, "") // remove containing dir
        .replace(/\.npy$/, ""); // remove npy extension
      // On windows replace backwards slash with forward slash.
      if (process.platform === "win32") {
        name = name.replace(/\\/g, "/");
      }
      assert(!path.isAbsolute(name));
      const tensor = await npy.load(fn);
      params.set(name, tensor);
      console.log(name, tensor.dtype, tensor.shape);
    }
    this.step_ = step;
    this.currentParams = params;
    return params;
  }

  async checkpoints(): Promise<number[]> {
    const out: number[] = [];
    for (const fn of fs.readdirSync(this.dir)) {
      if (fn.match(/^\d+$/)) {
        out.push(+fn);
      }
    }
    return out.sort((a, b) => b - a);
  }

  async deleteCheckpoint(step: number): Promise<void> {
    const p = this.checkpointPath(step);
    rimraf.sync(p);
    await print("Checkpoint deleted", p);
  }

  private checkpointPath(step: number): string {
    return path.normalize(path.join(this.dir, String(step).padStart(8, "0")));
  }

  // Saves a checkpoint to disk.
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
    await print(`Checkpoint saved ${megs} mb`, checkpointPath);
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
