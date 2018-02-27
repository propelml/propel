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

import * as path from "path";
import { test } from "../tools/tester";
import { imread } from "./image";
import { assert, assertShapesEqual } from "./util";

// The tests use these files:
// fa57d083e48e999ed3f210aefd92e5f7  testdata/sample.png
const pngPath = path.join(__dirname, "testdata/sample.png");
// d37ff170a58223de46152ce837b8e0c4  testdata/sample.jpg
const jpgPath = path.join(__dirname, "testdata/sample.jpg");

test(async function readPNG() {
  const img = await imread(pngPath);
  assertShapesEqual([4, 64, 64], img.shape);
  const data = img.dataSync();
  assert(data[0] === 255);
  assert(data[34] === 125);
  assert(data[3254] === 123);
});

test(async function readJPEG() {
  const img = await imread(jpgPath);
  assertShapesEqual([4, 64, 64], img.shape);
  const data = img.dataSync();
  assert(data[0] === 48);
  assert(data[34] === 45);
  assert(data[3254] === 191);
});

test(async function readRGBPNG() {
  const img = await imread(pngPath, "RGB");
  assertShapesEqual([3, 64, 64], img.shape);
  const data = img.dataSync();
  assert(data[0] === 255);
  assert(data[34] === 125);
  assert(data[3254] === 123);

});

test(async function readRGBJPEG() {
  const img = await imread(jpgPath, "RGB");
  assertShapesEqual([3, 64, 64], img.shape);
  const data = img.dataSync();
  assert(data[0] === 48);
  assert(data[34] === 45);
  assert(data[3254] === 191);
});
