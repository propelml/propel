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

import { test } from "../tools/tester";
import { imread } from "./im";
import { assert, assertShapesEqual, fetch2ArgManipulation,
  IS_NODE } from "./util";

// The tests use these files:
// fa57d083e48e999ed3f210aefd92e5f7  testdata/sample.png
const pngPath = fetch2ArgManipulation("src/testdata/sample.png");
// d37ff170a58223de46152ce837b8e0c4  testdata/sample.jpg
const jpgPath = fetch2ArgManipulation("src/testdata/sample.jpg");

test(async function im_pngEncoder() {
  const img = await imread(pngPath);
  assertShapesEqual([4, 64, 64], img.shape);
  const data = img.dataSync();
  assert(data[0] === 255);
  assert(data[34] === 125);
  assert(data[3254] === 123);
});

test(async function im_jpegEncoder() {
  const img = await imread(jpgPath);
  assertShapesEqual([4, 64, 64], img.shape);
  const data = img.dataSync();
  // JPEG-js does not work properly on Node.js and we're aware of this bug
  // so skip these tests for Node.js
  assert(data[0] === 49 || IS_NODE);
  assert(data[34] === 41 || IS_NODE);
  assert(data[3254] === 192 || IS_NODE);
});

test(async function im_toRGB() {
  const img = await imread(pngPath, "RGB");
  assertShapesEqual([3, 64, 64], img.shape);
  const data = img.dataSync();
  assert(data[0] === 255);
  assert(data[34] === 125);
  assert(data[3254] === 123);
});

test(async function im_toGrayscale() {
  const img = await imread(pngPath, "L");
  assertShapesEqual([1, 64, 64], img.shape);
  const data = img.dataSync();
  assert(data.length === 64 * 64);
});
