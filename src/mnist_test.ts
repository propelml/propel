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
import { assert, assertAllEqual, assertShapesEqual } from "./util";

test(async function mnist_trainSplit() {
  const dataset = mnist.load("train", 256, false);
  const {images, labels} = await dataset.next();

  assertShapesEqual(images.shape, [256, 28, 28]);
  // TODO Ultimately the mnist dataset should return uint8 tensors for the
  // images. However due to a performance workaround, we are using float32
  // for now.
  assert(images.dtype === "float32");
  assertAllEqual(images.reduceMax(), 255);
  assertAllEqual(images.reduceMean(), 32.7086296081543);

  assertShapesEqual(labels.shape, [256]);
  // assert(labels.dtype === "uint8");
  // > hexdump deps/mnist/train-labels-idx1-ubyte | head -1
  // 0000000 00 00 08 01 00 00 ea 60 05 00 04 01 09 02 01 03
  //         ---magic--- ---items--- ---------labels--------
  assertAllEqual(labels.slice([0], [8]), [5, 0, 4, 1, 9, 2, 1, 3]);
  console.log("TrainSplit ok");
});

test(async function mnist_testSplit() {
  const dataset = mnist.load("test", 16, false);
  const {images, labels} = await dataset.next();
  assertShapesEqual(images.shape, [16, 28, 28]);
  // assert(images.dtype === "uint8");
  assertShapesEqual(labels.shape, [16]);
  // assert(labels.dtype === "uint8");
  assertAllEqual(labels.slice([0], [3]), [7, 2, 1]);
  console.log("TestSplit ok");
});
