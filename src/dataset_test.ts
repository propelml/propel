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
import { assert, assertAllClose, assertAllEqual, assertShapesEqual }
  from "./tensor_util";
import { IS_NODE } from "./util";

test(async function dataset_datasetFromSlices() {
  const labels = pr.tensor([
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
  const ds = dataset.dataset("mnist/train").batch(5);
  const { images, labels } = await ds.next();
  assertAllEqual(labels, [ 5, 0, 4, 1, 9 ]);
  assertShapesEqual(images.shape, [5, 28, 28]);
});

test(async function dataset_repeatSlices() {
  const labels = pr.tensor([
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

test(async function dataset_loadIris() {
  const { features, labels } = await dataset.loadIris();
  assertShapesEqual(features.shape, [150, 4]);
  assertShapesEqual(labels.shape, [150]);
});

// Test accessing the iris dataset through the dataset API.
test(async function dataset_iris() {
  const ds = dataset.dataset("iris").batch(2);
  const { features, labels } = await ds.next();
  assertAllClose(features.slice([0, 0], [2, -1]),
                 [[ 5.1,  3.5,  1.4,  0.2], [ 4.9,  3. ,  1.4,  0.2]]);
  assertAllEqual(labels.slice([0], [2]), [0, 0]);
});

test(async function dataset_breastCancer() {
  const ds = dataset.dataset("breast_cancer").batch(2);
  const { features, labels } = await ds.next();
  const firstFeatures = features.slice([0, 0], [2, 4]);
  assertAllClose(firstFeatures,
    [[   17.99,    10.38,   122.8 ,  1001.  ],
     [   20.57,    17.77,   132.9 ,  1326.  ]]);
  assertAllEqual(labels.slice([0], [2]), [0, 0]);
});

test(async function dataset_wine() {
  const ds = dataset.dataset("wine").batch(2);
  const { features, labels } = await ds.next();
  const firstFeatures = features.slice([0, 0], [2, 4]);
  assertAllClose(firstFeatures,
    [[ 14.23,   1.71,   2.43,  15.6 ],
     [ 13.2 ,   1.78,   2.14,  11.2 ]]);
  assertAllEqual(labels.slice([0], [2]), [0, 0]);
});

test(async function dataset_iterable() {
  const ds = dataset.dataset("iris").batch(2);
  const labelCounts = [0, 0, 0];
  for (const p of ds) {
    const { labels } = await p;
    const labelsD = labels.dataSync();
    labelCounts[labelsD[0]]++;
    labelCounts[labelsD[1]]++;
  }
  assertAllEqual(labelCounts, [50, 50, 50]);
});

test(async function dataset_iterableEndCondition() {
  const features = pr.tensor([
    [ 0, 1, 0 ],
    [ 1, 0, 0 ],
    [ 0, 0, 1 ],
  ]);
  const ds = dataset.datasetFromSlices({ features }).batch(2).repeat(4);
  let count = 0;
  for (const p of ds) {
    const { features } = await p;
    count += features.shape[0];
  }
  assert(count === 12);
});

test(async function dataset_shuffleSmoke() {
  const labels = pr.tensor([
    [ 0, 1, 0 ],
    [ 1, 0, 0 ],
    [ 0, 0, 1 ],
  ]);
  const ds = dataset.datasetFromSlices({ labels }).shuffle(2);
  const el0 = await ds.next();
  assert(el0 != null);
  const el1 = await ds.next();
  assert(el1 != null);
  const el2 = await ds.next();
  assert(el2 != null);
  const el3 = await ds.next();
  assert(el3 == null);
});

test(async function dataset_cifar10() {
  // Only test on Node because we have no cache in the browser
  // and so it will download the whole cifar10 train set each
  // time.
  if (IS_NODE) {
    const ds = await dataset.dataset("cifar10/train").batch(4);
    const { images, labels } = await ds.next();
    assertShapesEqual(images.shape, [4, 32, 32, 3]);
    assertShapesEqual(labels.shape, [4]);
    assertAllEqual(images.slice([0, 0, 0, 0], [1, 5, 1, 1]).squeeze(),
      [59, 33, 97, 154, 142]);
    assertAllEqual(labels, [6, 9, 9, 4]);
  }
});
