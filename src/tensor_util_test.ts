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
import { assertShapesEqual, bcastGradientArgs } from "./tensor_util";

test(async function tensor_util_bcastGradientArgs() {
  let [r0, r1] = bcastGradientArgs([2, 3, 5], [1]);
  assertShapesEqual(r0, []);
  assertShapesEqual(r1, [0, 1, 2]);

  [r0, r1] = bcastGradientArgs([1], [2, 3, 5]);
  assertShapesEqual(r0, [0, 1, 2]);
  assertShapesEqual(r1, []);

  [r0, r1] = bcastGradientArgs([2, 3, 5], [5]);
  assertShapesEqual(r0, []);
  assertShapesEqual(r1, [0, 1]);

  [r0, r1] = bcastGradientArgs([5], [2, 3, 5]);
  assertShapesEqual(r0, [0, 1]);
  assertShapesEqual(r1, []);

  [r0, r1] = bcastGradientArgs([2, 3, 5], [3, 5]);
  assertShapesEqual(r0, []);
  assertShapesEqual(r1, [0]);

  [r0, r1] = bcastGradientArgs([3, 5], [2, 3, 5]);
  assertShapesEqual(r0, [0]);
  assertShapesEqual(r1, []);

  [r0, r1] = bcastGradientArgs([2, 3, 5], [3, 1]);
  assertShapesEqual(r0, []);
  assertShapesEqual(r1, [0, 2]);

  [r0, r1] = bcastGradientArgs([3, 1], [2, 3, 5]);
  assertShapesEqual(r0, [0, 2]);
  assertShapesEqual(r1, []);

  [r0, r1] = bcastGradientArgs([2, 1, 5], [3, 1]);
  assertShapesEqual(r0, [1]);
  assertShapesEqual(r1, [0, 2]);

  [r0, r1] = bcastGradientArgs([3, 1], [2, 1, 5]);
  assertShapesEqual(r0, [0, 2]);
  assertShapesEqual(r1, [1]);
});
