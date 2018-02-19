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
import * as pr from "./api";
import * as dataset from "./dataset";
import { assert, assertAllEqual, assertShapesEqual } from "./util";

test(async function dataset_datasetFromSlices() {
  const labels = pr.T([
    [ 0, 1, 0 ],
    [ 1, 0, 0 ],
    [ 0, 0, 1 ],
  ]);
  let ds = dataset.datasetFromSlices({ labels });

  // XXX should shape be [1,3] or [3] ?
  let el = await ds.next();
  assertAllEqual(el.labels, [[ 0, 1, 0 ]]);
  el = await ds.next();
  assertAllEqual(el.labels, [[ 1, 0, 0 ]]);
  el = await ds.next();
  assertAllEqual(el.labels, [[ 0, 0, 1 ]]);
  el = await ds.next();
  assert(el == null);

  // Try with batch(2)
  ds = dataset.datasetFromSlices({ labels }).batch(2);
  el = await ds.next();
  assertAllEqual(el.labels, [
    [ 0, 1, 0 ],
    [ 1, 0, 0 ],
  ]);
  el = await ds.next();
  assertAllEqual(el.labels, [
    [ 0, 0, 1 ],
  ]);
  el = await ds.next();
  assert(el == null);
});

test(async function dataset_mnist() {
  const ds = dataset.dataset("mnist").batch(5);
  const { images, labels } = await ds.next();
  assertAllEqual(labels, [ 5, 0, 4, 1, 9 ]);
  assertShapesEqual(images.shape, [5, 28, 28]);
});

test(async function dataset_repeatSlices() {
  const labels = pr.T([
    [ 0, 1, 0 ],
    [ 1, 0, 0 ],
    [ 0, 0, 1 ],
  ]);
  const ds = dataset.datasetFromSlices({ labels }).repeat(2);

  let el = await ds.next();
  assertAllEqual(el.labels, [[ 0, 1, 0 ]]);
  el = await ds.next();
  assertAllEqual(el.labels, [[ 1, 0, 0 ]]);
  el = await ds.next();
  assertAllEqual(el.labels, [[ 0, 0, 1 ]]);
  el = await ds.next();
  assertAllEqual(el.labels, [[ 0, 1, 0 ]]);
  el = await ds.next();
  assertAllEqual(el.labels, [[ 1, 0, 0 ]]);
  el = await ds.next();
  assertAllEqual(el.labels, [[ 0, 0, 1 ]]);
  el = await ds.next();
  assert(el == null);
});
