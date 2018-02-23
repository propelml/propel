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
import * as mnist from "./mnist";
import { assertAllEqual, assertShapesEqual } from "./util";

test(async function mnist_trainSplit() {
  const { images, labels } = await mnist.loadSplit("train");
  assertShapesEqual(images.shape, [60000, 28, 28]);
  assertAllEqual(images.reduceMax(), 255);
  assertShapesEqual(labels.shape, [60000]);
  // assert(labels.dtype === "uint8");
  // > hexdump deps/mnist/train-labels-idx1-ubyte | head -1
  // 0000000 00 00 08 01 00 00 ea 60 05 00 04 01 09 02 01 03
  //         ---magic--- ---items--- ---------labels--------
  assertAllEqual(labels.slice([0], [8]), [5, 0, 4, 1, 9, 2, 1, 3]);
});

test(async function mnist_testSplit() {
  const {images, labels} = await mnist.loadSplit("test");
  assertShapesEqual(images.shape, [10000, 28, 28]);
  assertShapesEqual(labels.shape, [10000]);
  assertAllEqual(labels.slice([0], [3]), [7, 2, 1]);
});
