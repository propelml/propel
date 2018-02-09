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
import * as util from "./util";
import { assert, assertEqual, assertShapesEqual } from "./util";

test(async function util_counterMap() {
  const m = new util.CounterMap();
  assertEqual(m.get(0), 0);
  m.inc(0);
  m.inc(0);
  assertEqual(m.get(0), 2);
  assertEqual(m.get(1), 0);
  m.inc(1);
  assertEqual(m.get(1), 1);
  m.dec(0);
  m.inc(1);
  assertEqual(m.get(0), 1);
  assertEqual(m.get(1), 2);
  const k = m.keys();
  console.log(k);
});

test(async function util_deepCloneArray() {
  const arr1 = [[1, 2], [3, 4]];
  const arr2 = util.deepCloneArray(arr1);
  // Verify that arrays have different identity.
  assert(arr1 !== arr2);
  assert(arr1[0] !== arr2[0]);
  assert(arr1[1] !== arr2[1]);
  // Verify that primitives inside the array have the same value.
  assert(arr1[0][0] === arr2[0][0]);
  assert(arr1[1][0] === arr2[1][0]);
  assert(arr1[0][1] === arr2[0][1]);
  assert(arr1[1][1] === arr2[1][1]);
});

test(async function util_bcastGradientArgs() {
  let [r0, r1] = util.bcastGradientArgs([2, 3, 5], [1]);
  assertShapesEqual(r0, []);
  assertShapesEqual(r1, [0, 1, 2]);

  [r0, r1] = util.bcastGradientArgs([1], [2, 3, 5]);
  assertShapesEqual(r0, [0, 1, 2]);
  assertShapesEqual(r1, []);

  [r0, r1] = util.bcastGradientArgs([2, 3, 5], [5]);
  assertShapesEqual(r0, []);
  assertShapesEqual(r1, [0, 1]);

  [r0, r1] = util.bcastGradientArgs([5], [2, 3, 5]);
  assertShapesEqual(r0, [0, 1]);
  assertShapesEqual(r1, []);

  [r0, r1] = util.bcastGradientArgs([2, 3, 5], [3, 5]);
  assertShapesEqual(r0, []);
  assertShapesEqual(r1, [0]);

  [r0, r1] = util.bcastGradientArgs([3, 5], [2, 3, 5]);
  assertShapesEqual(r0, [0]);
  assertShapesEqual(r1, []);

  [r0, r1] = util.bcastGradientArgs([2, 3, 5], [3, 1]);
  assertShapesEqual(r0, []);
  assertShapesEqual(r1, [0, 2]);

  [r0, r1] = util.bcastGradientArgs([3, 1], [2, 3, 5]);
  assertShapesEqual(r0, [0, 2]);
  assertShapesEqual(r1, []);

  [r0, r1] = util.bcastGradientArgs([2, 1, 5], [3, 1]);
  assertShapesEqual(r0, [1]);
  assertShapesEqual(r1, [0, 2]);

  [r0, r1] = util.bcastGradientArgs([3, 1], [2, 1, 5]);
  assertShapesEqual(r0, [0, 2]);
  assertShapesEqual(r1, [1]);
});
