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
import * as path from "path";
import { assert } from "./util";

export function isDir(p: string): boolean {
  try {
    return fs.statSync(p).isDirectory();
  } catch (e) {
    if (e.code === "ENOENT") return false;
    throw e;
  }
}

/** Returns "$HOME/.propel/" or PROPEL_DIR env var. */
export function propelDir(): string {
  if (process.env.PROPEL_DIR) {
    return process.env.PROPEL_DIR;
  } else {
    const homeDir = process.platform === "win32" ? process.env.USERPROFILE
                                                 : process.env.HOME;
    return path.join(homeDir, ".propel/");
  }
}

/** Recursive mkdir. */
export function mkdirp(dirname: string): void {
  if (!isDir(dirname)) {
    const parentDir = path.dirname(dirname);
    assert(parentDir !== dirname && parentDir.length > 1);
    mkdirp(parentDir);
    fs.mkdirSync(dirname);
  }
}
