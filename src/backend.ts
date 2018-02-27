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

// BasicTensor abstracts TensorFlow and DeepLearn BackendOps. Operations on
// BasicTensors are not traced in backprop and the class is not exposed to the
// public API.

import * as dl from "./dl";
import * as tf from "./tf";
import * as types from "./types";
import { deepCloneArray, flatten, inferShape, IS_WEB, isTypedArray,
  makeTypedArray } from "./util";

// These globals will be set by onLoad
export let tensorClass: any;
export let backend: string;
export let bo: types.BackendOps;

let onLoadCalled = false;
(function onLoad() {
  if (onLoadCalled) {
    console.warn("Warning: backend.onLoad called more than once.");
    return;
  }
  onLoadCalled = true;

  if (preferTF() && tf.loadBinding()) {
    console.log("Using TF backend.");
    tensorClass = tf.TensorTF;
    bo = new tf.OpsTF();
    backend = "tf";
  } else {
    console.log("Using DL backend.");
    tensorClass = dl.TensorDL;
    bo = new dl.OpsDL();
    backend = "dl";
  }
})();

function preferTF(): boolean {
  // If we're in the browser, don't even attempt it.
  if (IS_WEB) return false;

  // This is to set the backend to either web or tensorflow.
  // Use this on the command line:
  //   PROPEL=web node myprogram.js
  // This is used in tools/test.js to run tests on both backends.
  if (!process.env.PROPEL || process.env.PROPEL === "tf") {
     // continue
  } else if (process.env.PROPEL === "dl") {
    return false;
  } else {
    throw Error("Bad value for env var PROPEL.");
  }
  return true;
}

export function convertBasic(x: types.TensorLike,
                             opts?: types.TensorOpts): types.BasicTensor {
  // Alias fromTypedArray for brevity.
  const create = tensorClass.fromTypedArray;

  const dtype = opts ? opts.dtype : undefined;
  const device = (opts ? opts.device : null) || "CPU:0";

  if (typeof x === "number") {
    // TODO On TF we should take advantage of createSmallHandle for scalars.
    return create(makeTypedArray([x], dtype), [], dtype, device);
  } else if (isTypedArray(x)) {
    return create(x, [x.length], dtype, device);
  } else if (Array.isArray(x)) {
    if (!(x instanceof Array)) {
      // Unfortunately deeplearnjs gets confused by an out-of-context array.
      // Therefore clone the array.
      x = deepCloneArray(x);
    }
    const shape = inferShape(x);
    const data = flatten(x) as number[];
    return create(makeTypedArray(data, dtype), shape, dtype, device);
  }
  throw new Error("Unreachable");
}
