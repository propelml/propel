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
import * as api from "./api";
import * as layers from "./layers";
import {
  assert,
  assertAllClose,
  assertShapesEqual,
} from "./tensor_util";

test(async function layer_conv2d() {
  const x = api.ones([1, 4, 4, 1]);
  const p = api.params();
  const out = layers.conv2d(x, p.scope("L1"), 2);
  assertShapesEqual(out.shape, [1, 4, 4, 2]);
  assert(p.has("L1/filter"));
  assert(p.has("L1/bias"));
  assertShapesEqual(p.get("L1/filter").shape, [3, 3, 1, 2]);
  assertShapesEqual(p.get("L1/bias").shape, [2]);
});

test(async function layer_linear() {
  const p = api.params();
  const x = api.zeros([2, 5]);
  const out = layers.linear(x, p.scope("L2"), 10);
  assert(p.has("L2/weights"));
  assert(p.has("L2/bias"));
  assertShapesEqual(out.shape, [2, 10]);
});

test(async function layer_batchNorm() {
  const p = api.params();
  const c0 = api.randn([2, 50, 50, 1]).mul(3).sub(7);
  const c1 = api.randn([2, 50, 50, 1]).mul(13).add(2);
  const x = c0.concat(3, c1);
  const out = layers.batchNorm(x, p.scope("bn"));
  assert(p.has("bn/mean"));
  assert(p.has("bn/variance"));
  assertAllClose(p.get("bn/mean"), [-7, 2], 1);
  assertAllClose(p.get("bn/variance"), [3 * 3, 13 * 13], 3);
  const m = out.moments([0, 1, 2]);
  assertAllClose(m.mean, [0, 0], 0.1);
  assertAllClose(m.variance, [1, 1], 0.1);
});
