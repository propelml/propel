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

// In Propel we define a layer to be any parametric operation. An op
// that uses Params. So linear(), for example, defines a weight and bias tensor
// inside the provided params object. Therefore it belongs in this module,
// layers. Our usage of "layers" diverges slightly from the nomenclature used
// in TensorFlow, where things like max pooling are also considered layers:
// tslint:disable-next-line:max-line-length
// https://github.com/tensorflow/tensorflow/blob/f73d7c90ed05bcf9f36f6a3be0c29efa5fef0f6e/tensorflow/contrib/layers/python/layers/layers.py#L77
// In Propel because maxPool is non-parametric, it is relegated to hang off of
// Tensor.
//
// conv2d is an unfortunate case: There is both the basic conv2d op, in which
// the user provides a filter, and the more commonly used conv2d layer - where
// params is used. Due to conflicting names, the basic conv2d has been put in
// api.ts instead of tensor.ts.

import * as ops from "./ops";
import { Params } from "./params";
import { Tensor } from "./tensor";
import {
  assert,
  assertShapesEqual,
} from "./tensor_util";
import * as types from "./types";

export interface LinearOpts {
  bias?: boolean;
  scale?: number;
}

// Slightly different than types.ConvOpts, this one includes "size".
export interface ConvOpts {
  size?: number | [number, number];
  stride?: number | [number, number];
  padding?: types.Padding;
  bias?: boolean;
}

const convDefaults: ConvOpts = {
  size: 3,
  stride: 1,
  padding: "same",
  bias: true,
};

export interface BatchNormOpts {
  decay: number;
  epsilon: number;
}

const bnDefaults = {
  decay: 0.997,
  epsilon: 1e-5,
};

export function linear(input: Tensor, params: Params, outDim: number,
                       { bias = true, scale = 0.01 }: LinearOpts = {}): Tensor {
  assert(input.rank >= 2);
  const p = params;
  let x = input;
  // Partially flatten tensor if needed.
  x = x.rank === 2 ? x : x.reshape([x.shape[0], -1]);
  const inDim = x.shape[x.rank - 1];
  const w = p.define("weights", () => ops.randn([inDim, outDim]).mul(scale));
  x = x.matmul(w);
  if (bias) {
    const b = p.define("bias", () => ops.zeros([outDim]));
    x = x.add(b);
  }
  return x;
}

export function conv2d(input: Tensor, params: Params, outChans,
                       opts?: ConvOpts): Tensor {
  let x = input;
  assert(x.rank === 4);
  opts = Object.assign(convDefaults, opts);
  const filter = params.define("filter", () =>
    ops.randn([
      opts.size,
      opts.size,
      x.shape[3],
      outChans
    ]));
  x = ops.conv2d(x, filter, opts);
  if (opts.bias) {
    const b = params.define("bias", () => ops.zeros([outChans]));
    x = x.add(b);
  }
  return x;
}

export function batchNorm(input: Tensor, params: Params,
                          opts?: BatchNormOpts): Tensor {
  opts = Object.assign(bnDefaults, opts);
  const p = params;
  const x = input;
  assert(x.rank === 4);
  const c = x.shape[3];
  let moments;
  let mean = p.define("mean", () => {
    moments = x.moments([0, 1, 2]);
    return moments.mean;
  });
  let variance = p.define("variance", () => {
    assert(moments);
    return moments.variance;
  });
  const ema = (newValue, oldValue) =>
    newValue.mul(opts.decay).add(oldValue.mul(1 - opts.decay));
  if (moments == null || p.isTraining) {
    moments = x.moments([0, 1, 2]);
    mean = p.set("mean", ema(moments.mean, mean));
    variance = p.set("variance", ema(moments.variance, variance));
  }
  assertShapesEqual(mean.shape, [c]);
  assertShapesEqual(variance.shape, [c]);
  return x.sub(mean).div(variance.add(opts.epsilon).sqrt());
}
