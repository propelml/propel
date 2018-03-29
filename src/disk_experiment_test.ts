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
import { DiskExperiment } from "./disk_experiment";
import { assert, assertAllEqual, assertEqual } from "./tensor_util";
import { process } from "./util";
import { isDir } from "./util_node";

function setup() {
  process.env.PROPEL_ROOT = path.join(os.tmpdir(), "propel_test");
  console.log("PROPEL_ROOT", process.env.PROPEL_ROOT);
  if (fs.existsSync(process.env.PROPEL_ROOT)) {
    console.log("rm -rf PROPEL_ROOT", process.env.PROPEL_ROOT);
    rimraf.sync(process.env.PROPEL_ROOT);
    fs.mkdirSync(process.env.PROPEL_ROOT);
  }
}

test(async function disk_experiment_saveRestore() {
  setup();
  const exp = new DiskExperiment("exp1");
  await exp.createOrRestore();
  const expDir = path.join(process.env.PROPEL_ROOT, "exp1");
  assert(isDir(expDir));
  assertEqual(fs.readdirSync(expDir).length, 0);
  const checkpoints = await exp.checkpoints();
  assertEqual(checkpoints.length, 0);
  assertEqual(exp.params.length, 0);

  exp.params.define("hello/world", () => [1, 2, 3]);
  await exp.save();
  const checkpointDirs = fs.readdirSync(expDir);
  assertEqual(checkpointDirs.length, 1);
  const checkpointDir = path.join(expDir, checkpointDirs[0]);
  assert(isDir(checkpointDir));
  assert(fs.existsSync(path.join(checkpointDir, "hello/world.npy")));

  // Try to load the checkpoint we just saved.
  const exp_ = new DiskExperiment("exp1");
  await exp_.createOrRestore();
  assertEqual(exp_.params.length, 1);
  const t = exp_.params.get("hello/world");
  assertAllEqual(t, [1, 2, 3]);
});
