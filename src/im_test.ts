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
import { propelURL } from "./fetch";
import { imread, toUint8Image } from "./im";
import { assertAllEqual } from "./tensor_util";
import { assert, assertShapesEqual } from "./tensor_util";
import { IS_NODE } from "./util";

// The tests use these files:
// fa57d083e48e999ed3f210aefd92e5f7  testdata/sample.png
const pngPath = propelURL + "src/testdata/sample.png";
// d37ff170a58223de46152ce837b8e0c4  testdata/sample.jpg
const jpgPath = propelURL + "src/testdata/sample.jpg";

function getPixel(channels, data, x, y, rgba = false) {
  // we know sample images are 64x64
  const idx = (rgba ? 4 : channels) * (y * 64 + x);
  const pixel = new Array(channels)
    .fill(null)
    .map((x, i) => data[idx + i]);
  return pixel;
}

test(async function im_pngEncoder() {
  const img = await imread(pngPath);
  assertShapesEqual(img.shape, [64, 64, 4]);
  const data = img.dataSync();
  assertAllEqual(getPixel(4, data, 2, 3), [255, 255, 255, 255]);
  assertAllEqual(getPixel(4, data, 20, 20), [242, 232, 237, 255]);
  assertAllEqual(getPixel(4, data, 62, 3), [255, 255, 255, 255]);
  assertAllEqual(getPixel(4, data, 34, 5), [120, 100, 169, 255]);
});

test(async function im_jpegEncoder() {
  const img = await imread(jpgPath);
  assertShapesEqual(img.shape, [64, 64, 4]);
  // JPEG-js does not work properly on Node.js and we're aware of this bug
  // so skip these tests for Node.js
  if (IS_NODE) {
    return;
  }
  const data = img.dataSync();
  assert(data[0] === 49);
  assert(data[4 * 34] === 41);
  assert(data[4 * 3254] === 192);
});

test(async function im_toRGB() {
  const img = await imread(pngPath, "RGB");
  assertShapesEqual(img.shape, [64, 64, 3]);
  const data = img.dataSync();
  assertAllEqual(getPixel(3, data, 2, 3), [255, 255, 255]);
  assertAllEqual(getPixel(3, data, 20, 20), [242, 232, 237]);
  assertAllEqual(getPixel(3, data, 62, 3), [255, 255, 255]);
  assertAllEqual(getPixel(3, data, 34, 5), [120, 100, 169]);
});

test(async function im_toGrayscale() {
  const img = await imread(pngPath, "L");
  assertShapesEqual(img.shape, [64, 64, 1]);
  const data = img.dataSync();
  assert(data.length === 64 * 64);
  assertAllEqual(getPixel(1, data, 2, 3), [255]);
  assertAllEqual(getPixel(1, data, 20, 20), [237]);
  assertAllEqual(getPixel(1, data, 62, 3), [255]);
});

test(async function im_toUint8ImageRGBA() {
  const img = await imread(pngPath);
  const rawImage = toUint8Image(img);
  assert(rawImage.width === 64);
  assert(rawImage.height === 64);
  assert(rawImage.data.length === 64 * 64 * 4);
  const data = rawImage.data;
  assertAllEqual(getPixel(4, data, 2, 3, true), [255, 255, 255, 255]);
  assertAllEqual(getPixel(4, data, 20, 20, true), [242, 232, 237, 255]);
  assertAllEqual(getPixel(4, data, 62, 3, true), [255, 255, 255, 255]);
  assertAllEqual(getPixel(4, data, 34, 5, true), [120, 100, 169, 255]);
});

test(async function im_toUint8ImageRGB() {
  const img = await imread(pngPath, "RGB");
  const rawImage = toUint8Image(img);
  assert(rawImage.width === 64);
  assert(rawImage.height === 64);
  assert(rawImage.data.length === 64 * 64 * 4);
  const data = rawImage.data;
  assertAllEqual(getPixel(3, data, 2, 3, true), [255, 255, 255]);
  assertAllEqual(getPixel(3, data, 20, 20, true), [242, 232, 237]);
  assertAllEqual(getPixel(3, data, 62, 3, true), [255, 255, 255]);
  assertAllEqual(getPixel(3, data, 34, 5, true), [120, 100, 169]);
});

test(async function im_toUint8ImageGrayscale() {
  const img = await imread(pngPath, "L");
  const rawImage = toUint8Image(img);
  assert(rawImage.width === 64);
  assert(rawImage.height === 64);
  assert(rawImage.data.length === 64 * 64 * 4);
  const data = rawImage.data;
  assertAllEqual(getPixel(1, data, 2, 3, true), [255]);
  assertAllEqual(getPixel(1, data, 20, 20, true), [237]);
  assertAllEqual(getPixel(1, data, 62, 3, true), [255]);
});
