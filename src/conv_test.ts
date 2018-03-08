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
import { range, Tensor } from "./api";
import { backend, bo } from "./backend";
import { cases, ConvTestCase } from "./conv_testcases";
import { assertAllClose } from "./tensor_util";
import { ConvOpts, DType, Shape } from "./types";

const format = "NHWC";

// Forward pass tests.
defineTests("fw", (c: ConvTestCase): Tensor => {
  const input = testInput("float32", c.inputShape);
  const filter = testInput("float32", c.filterShape);
  return input.conv2d(filter, {
    strides: c.strides,
    padding: c.padding,
    format,
  });
});

// Backwards pass tests thru the filter.
defineTests("bwFilter", (c: ConvTestCase): Tensor => {
  const input = testInput("float32", c.inputShape);
  const gradient = testInput("float32", c.outputShape);
  const opts: ConvOpts = {
    strides: c.strides,
    padding: c.padding,
    format,
  };
  const b = bo.conv2dBackpropFilter(gradient.storage, input.storage,
                                    c.filterShape, opts);
  return new Tensor(b);
});

// Backwards pass tests thru the input..
defineTests("bwInput", (c: ConvTestCase): Tensor => {
  const filter = testInput("float32", c.filterShape);
  const gradient = testInput("float32", c.outputShape);
  const opts: ConvOpts = {
    strides: c.strides,
    padding: c.padding,
    format,
  };
  const b = bo.conv2dBackpropInput(gradient.storage, c.inputShape,
                                   filter.storage, opts);
  return new Tensor(b);
});

function defineTests(suite: string,
                     fn: (c: ConvTestCase) => Tensor) {
  for (const c of cases[suite]) {
    test({
      fn: () => {
        if (c.skip && c.skip.indexOf(backend) >= 0) {
          console.log(`Skip ${c.name}. skip = "${c.skip}"`);
          return;
        }
        const actual = fn(c);
        assertAllClose(actual.dataSync(), c.expected);
      },
      name: `conv_${suite}_${c.name}`,
    });
  }
}

function testInput(dtype: DType, shape: Shape): Tensor {
  return range(1, prod(shape) + 1).cast(dtype).reshape(shape);
}

// Ideally this would be an api.ts function.
function prod(array: number[]): number {
  return array.reduce((a, b) => a * b);
}
