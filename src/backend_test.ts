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
import { bo, convertBasic } from "./backend";
import { assertAllClose, assertAllEqual, assertShapesEqual } from "./util";

const T = convertBasic;

test(async function backend_shapes() {
  const a = T([[1, 2], [3, 4]]);
  assertShapesEqual(a.shape, [2, 2]);

  const b = T([[[1, 2]]]);
  assertShapesEqual(b.shape, [1, 1, 2]);

  assertShapesEqual(T(42).shape, []);
  assertShapesEqual(T([42]).shape, [1]);
});

test(async function backend_mul() {
  const a = T([[1, 2], [3, 4]]);
  const expected = T([[1, 4], [9, 16]]);
  const actual = bo.mul(a, a);
  assertAllEqual(actual.shape, [2, 2]);
  assertAllEqual(actual, expected);
});

test(async function backend_square() {
  const a = T([[1, 2], [3, 4]]);
  const expected = T([[1, 4], [9, 16]]);
  const actual = bo.square(a);
  assertAllEqual(actual.shape, [2, 2]);
  assertAllEqual(actual, expected);
});

test(async function backend_cosh() {
  const a = T([[1, 2], [3, 4]]);
  const actual = bo.cosh(a);
  assertAllEqual(actual.shape, [2, 2]);
  const expected = [
    [  1.54308063,   3.76219569],
    [ 10.067662  ,  27.30823284],
  ];
  assertAllClose(actual, expected);
});

test(async function backend_reluGrad() {
  const grad = T([[-1, 42], [-3, 4]]);
  const ans = T([[-7, 4], [ 0.1, -9 ]]);
  const actual = bo.reluGrad(grad, ans);
  assertAllEqual(actual.shape, [2, 2]);
  assertAllClose(actual, [[0, 42], [-3,  0]]);
});
