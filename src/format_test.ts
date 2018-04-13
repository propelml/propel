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
import { float32, int32, range } from "./api";
import { toString } from "./format";
import { assertEqual } from "./util";

test(async function format_1DInt32() {
  let actual = toString(int32([1, 2, 3, 4]));
  let expected = "[1, 2, 3, 4]";
  assertEqual(actual, expected);

  // With negative values.
  actual = toString(int32([-2, -1, 0, 1]));
  expected = "[-2, -1, 0, 1]";
  assertEqual(actual, expected);
});

test(async function format_2DInt32() {
  const actual = toString(int32([[1, 2], [3, 4]]));
  const expected = "[[1, 2],\n [3, 4]]";
  assertEqual(actual, expected);
});

test(async function format_3DInt32() {
  const actual = toString(int32([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]));
  const expected = "[[[1, 2],\n  [3, 4]],\n\n [[5, 6],\n  [7, 8]]]";
  assertEqual(actual, expected);
});

test(async function format_1DFloat32() {
  let d = [1.2, 0., 0., 0.123];
  let actual = toString(float32(d));
  let expected = "[ 1.2  ,  0.   ,  0.   ,  0.123]";
  assertEqual(actual, expected);

  d = [5.1, 3.5, 1.4, 0.2];
  actual = toString(float32(d));
  expected = "[ 5.1,  3.5,  1.4,  0.2]";
  assertEqual(actual, expected);

  // With negative values.
  d = [-1, -0.5, 0, 0.5, 1];
  actual = toString(float32(d));
  expected = "[-1. , -0.5,  0. ,  0.5,  1. ]";
  assertEqual(actual, expected);
});

test(async function format_2DFloat32() {
  let actual = toString(float32([[1., 2.], [3., 4.]]));
  let expected = "[[ 1.,  2.],\n [ 3.,  4.]]";
  assertEqual(actual, expected);

  actual = toString(float32([[1.2, 0, 0, 0.1], [1.2, 0, 0.5, 0]]));
  expected = "[[ 1.2,  0. ,  0. ,  0.1],\n [ 1.2,  0. ,  0.5,  0. ]]";
  assertEqual(actual, expected);
 });

test(async function format_4DInt32() {
  const actual = toString(range(2 * 2 * 3 * 3).reshape([2, 2, 3, 3]));
  // tslint:disable:max-line-length
  const expected = "[[[[ 0,  1,  2],\n   [ 3,  4,  5],\n   [ 6,  7,  8]],\n\n  [[ 9, 10, 11],\n   [12, 13, 14],\n   [15, 16, 17]]],\n\n\n [[[18, 19, 20],\n   [21, 22, 23],\n   [24, 25, 26]],\n\n  [[27, 28, 29],\n   [30, 31, 32],\n   [33, 34, 35]]]]";
  assertEqual(actual, expected);
});

test(async function format_threshold() {
  const actual = toString(range(2000));
  const expected = "[   0,    1,    2, ..., 1997, 1998, 1999]";
  assertEqual(actual, expected);
});
