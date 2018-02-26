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
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as rimraf from "rimraf";
import { test } from "../tools/tester";
import { DiskExperiment, isDir } from "./disk_experiment";
import { assert } from "./util";

function setup() {
  process.env.PROPEL_DIR = path.join(os.tmpdir(), "propel_test");
  console.log("PROPEL_DIR", process.env.PROPEL_DIR);
  if (fs.existsSync(process.env.PROPEL_DIR)) {
    console.log("rm -rf PROPEL_DIR", process.env.PROPEL_DIR);
    rimraf.sync(process.env.PROPEL_DIR);
    fs.mkdirSync(process.env.PROPEL_DIR);
  }
}

test(async function disk_experiment_create() {
  setup();
  const exp = new DiskExperiment("exp1");
  await exp.createOrRestore();
  assert(isDir(path.join(process.env.PROPEL_DIR, "exp1")));
  const checkpoints = await exp.checkpoints();
  assert(checkpoints.length === 0);
});
