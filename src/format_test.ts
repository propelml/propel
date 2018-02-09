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
import { toString } from "./format";
import { assert } from "./util";

test(async function format_1DInt32() {
  let actual = toString([4], new Int32Array([1, 2, 3, 4]));
  let expected = "[1, 2, 3, 4]";
  assert(actual === expected);

  // With negative values.
  const d = [-2, -1, 0, 1];
  actual = toString([4], new Int32Array(d));
  expected = "[-2, -1, 0, 1]";
  assert(actual === expected);
});

test(async function format_2DInt32() {
  const actual = toString([2, 2], new Int32Array([1, 2, 3, 4]));
  const expected = "[[1, 2],\n [3, 4]]";
  console.log(actual);
  assert(actual === expected);
});

test(async function format_1DFloat32() {
  let d = [ 1.2,  0. ,  0. ,  0.123];
  let actual = toString([4], new Float32Array(d));
  let expected = "[ 1.2  ,  0.   ,  0.   ,  0.123]";
  console.log(actual);
  console.log(expected);
  assert(actual === expected);

  // With negative values.
  d = [-1, -0.5, 0, 0.5, 1];
  actual = toString([5], new Float32Array(d));
  expected = "[-1. , -0.5,  0. ,  0.5,  1. ]";
  console.log(actual);
  console.log(expected);
  assert(actual === expected);
});

test(async function format_2DFloat32() {
  let actual = toString([2, 2], new Float32Array([1, 2, 3, 4]));
  let expected = "[[ 1.,  2.],\n [ 3.,  4.]]";
  console.log(actual);
  assert(actual === expected);

  const ta = new Float32Array([1.2, 0, 0, 0.1, 1.2, 0, 0.5, 0]);
  actual = toString([2, 4], ta);
  expected = "[[ 1.2,  0. ,  0. ,  0.1],\n [ 1.2,  0. ,  0.5,  0. ]]";
  console.log(actual);
  assert(actual === expected);
});
