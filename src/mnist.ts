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
import { tensor, Tensor } from "./api";
import * as cache from "./cache";
import { assert } from "./util";

export function filenames(split: string): [string, string] {
  if (split === "train") {
    return [
      "http://propelml.org/data/mnist/train-labels-idx1-ubyte.bin",
      "http://propelml.org/data/mnist/train-images-idx3-ubyte.bin",
    ];
  } else if (split === "test") {
    return [
      "http://propelml.org/data/mnist/t10k-labels-idx1-ubyte.bin",
      "http://propelml.org/data/mnist/t10k-images-idx3-ubyte.bin",
    ];
  } else {
    throw new Error(`Bad split: ${split}`);
  }
}

function littleEndianToBig(val) {
  return ((val & 0x00FF) << 24) |
         ((val & 0xFF00) << 8) |
         ((val >> 8) & 0xFF00) |
         ((val >> 24) & 0x00FF);
}

export async function loadSplit(split: string):
    Promise<{images: Tensor, labels: Tensor}> {
  const [hrefLabels, hrefImages] = filenames(split);
  const imagesPromise = loadFile2(hrefImages);
  const labelsPromise = loadFile2(hrefLabels);
  const [images, labels] = await Promise.all([imagesPromise, labelsPromise]);
  return { images, labels };
}

async function loadFile2(href: string) {
  const ab = await cache.fetchWithCache(href);
  const i32 = new Int32Array(ab);
  const ui8 = new Uint8Array(ab);

  let i = 0;
  const magicValue = littleEndianToBig(i32[i++]);
  let isImages;
  if (magicValue === 2051) {
    isImages = true;
  } else if (magicValue === 2049) {
    isImages = false;
  } else {
    throw Error("Bad magic value.");
  }
  const numExamples = littleEndianToBig(i32[i++]);

  let t;
  if (isImages) {
    assert(littleEndianToBig(i32[i++]) === 28);
    assert(littleEndianToBig(i32[i++]) === 28);
    const tensorData = new Int32Array(ui8.slice(4 * i));
    t = tensor(tensorData, {dtype: "int32"});
  } else {
    const tensorData = new Int32Array(ui8.slice(4 * i));
    t = tensor(tensorData, {dtype: "int32"});
  }
  const shape = isImages ? [numExamples, 28, 28] : [numExamples];

  return t.reshape(shape);
}
