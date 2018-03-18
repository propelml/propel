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
// Deep Learn JS backend.

// These files call a top-level registerBackend, which must be run before doing
// any work with DL.
import "./dl/math/backends/backend_cpu";
import "./dl/math/backends/backend_webgl";

import { ENV } from "./dl/environment";
import { MathBackendWebGL }
 from "./dl/math/backends/backend_webgl";
import { NDArrayMath } from "./dl/math/math";
import { Array1D, Array2D, Array3D, Array4D, NDArray, Scalar }
  from "./dl/math/ndarray";
import { MatrixOrientation } from "./dl/math/types";
import { assert, getDType, makeTypedArray } from "./tensor_util";
import * as types from "./types";

const deviceRegistry = new Map<string, NDArrayMath>();
function lookupMath(device: string): NDArrayMath {
  assert(deviceRegistry.has(device));
  return deviceRegistry.get(device);
}

const cpuMath = new NDArrayMath("cpu");
ENV.setMath(cpuMath);
deviceRegistry.set("CPU:0", cpuMath);

let gpuMath;
const webglBackend = ENV.getBackend("webgl") as MathBackendWebGL;
if (webglBackend) {
  gpuMath = new NDArrayMath("webgl");
  deviceRegistry.set("GPU:0", gpuMath);
}

// TODO Propel largely follows the dtype scheme layed out by DL, but
// additonally adds the uint8 type. This function will map that down to int32.
// This is a hack and a potential source of bugs.
type DTypeDL = "float32" | "int32" | "bool";
function dtypeDL(propelDtype: types.DType): DTypeDL {
  switch (propelDtype) {
      case "int32":
      case "float32":
      case "bool":
        return propelDtype;
      case "uint8":
        return "int32";
  }
}

export type TensorDL = NDArray; // Note NDArray implements Storage.

export class OpsDL implements types.BackendOps {

  copyToDevice(x: TensorDL, device: string): TensorDL {
    const math = lookupMath(device);
    // TODO Don't use dataSync() here.
    return NDArray.make(x.shape, {values: x.dataSync()},
                        x.dtype, math);
  }

  getDevice(x: TensorDL): string {
    if (x.math === cpuMath) return "CPU:0";
    if (x.math === gpuMath) return "GPU:0";
    throw new Error("Unreachable");
  }

  listDevices(): string[] {
    const d = ["CPU:0"];
    if (webglBackend) d.push("GPU:0");
    return d;
  }

  fromTypedArray(values: types.TypedArray, shape: types.Shape,
                 dtype?: types.DType, device?: string): TensorDL {
    if (dtype == null) {
      dtype = getDType(values);
    }
    if (device == null) {
      device = "CPU:0";
    }
    const math = lookupMath(device);
    return NDArray.make(shape, { values }, dtype as any, math);
  }

  add(x: TensorDL, y: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.add(x, y);
  }

  sub(x: TensorDL, y: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.sub(x, y);
  }

  mul(x: TensorDL, y: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.multiply(x, y);
  }

  div(x: TensorDL, y: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.divide(x, y);
  }

  neg(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.neg(x);
  }

  exp(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.exp(x);
  }

  log(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.log(x);
  }

  matmul(x: TensorDL, y: TensorDL, transposeA = false,
         transposeB = false): TensorDL {
    ENV.setMath(x.math);
    const f = (t) => t ?
      MatrixOrientation.TRANSPOSED :
      MatrixOrientation.REGULAR;
    assert(x.shape.length === 2 && y.shape.length === 2);
    const x2 = x as Array2D;
    const y2 = y as Array2D;
    return x.math.matMul(x2, y2, f(transposeA), f(transposeB));
  }

  setDiag(x: TensorDL, diag: TensorDL): TensorDL {
    if (x.shape.length !== 2 || diag.shape.length !== 1) {
      throw new Error("Not implemented");
    }
    // DL doesn't support WebGL for this yet, so force CPU.
    ENV.setMath(cpuMath);
    return cpuMath.setDiag(x as Array2D, diag as Array1D);
  }

  onesLike(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ones = NDArray.zerosLike(x);
    ones.fill(1.0);
    assert(ones.math === x.math);
    return ones;
  }

  zerosLike(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const zeros = NDArray.zerosLike(x);
    assert(zeros.math === x.math);
    return zeros;
  }

  fill(value: TensorDL, shape: types.Shape): TensorDL {
    if (value.shape.length !== 0) {
      throw new Error("Fill value must be a scalar.");
    }
    ENV.setMath(value.math);
    const out = NDArray.zeros(shape, value.dtype);
    out.fill(value.dataSync()[0]);
    assert(out.math === value.math);
    return out;
  }

  square(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.square(x);
  }

  sqrt(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.sqrt(x);
  }

  pow(x: TensorDL, exponent: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.pow(x, exponent);
  }

  sin(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.sin(x);
  }

  cos(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.cos(x);
  }

  tan(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.tan(x);
  }

  sinh(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.sinh(x);
  }

  cosh(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.cosh(x);
  }

  tanh(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.tanh(x);
  }

  relu(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.relu(x);
  }

  reluGrad(grad: TensorDL, features: TensorDL): TensorDL {
    const m = grad.math;
    const s = m.step(features);
    return m.multiply(grad, s);
  }

  sigmoid(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.sigmoid(x);
  }

  abs(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.abs(x);
  }

  randn(shape: types.Shape, seed?: number): TensorDL {
    ENV.setMath(cpuMath);
    return NDArray.randNormal(shape, 0, 1, "float32", seed);
  }

  linspace(start: number, stop: number, num: number): TensorDL {
    const d = (stop - start) / (num - 1);
    const ta = new Float32Array(num);
    for (let i = 0; i <= num - 1; ++i) {
      ta[i] = start + i * d;
    }
    return this.fromTypedArray(ta, [num]);
  }

  range(start: number, limit: number, delta: number): TensorDL {
    const num = (limit - start) / delta;
    const ta = new Int32Array(num);
    for (let i = 0; i < num; ++i) {
      ta[i] = start + i * delta;
    }
    return this.fromTypedArray(ta, [num]);
  }

  transpose(x: TensorDL, perm: TensorDL): TensorDL {
    const permArr = Array.from(perm.dataSync());
    return x.math.transpose(x, permArr);
  }

  // TODO dims should not be a tensor.
  reverse(x: TensorDL, dims: TensorDL): TensorDL {
    const a = x;
    const dims_ = dims.dataSync();
    // TODO move to deeplearnjs/src/math/backends/backend_cpu.ts
    const resultValues = makeTypedArray(a.size, x.dtype);
    const values = a.dataSync();
    const dtype = dtypeDL(x.dtype);
    const result = NDArray.make(a.shape, {values: resultValues},
                                dtype, x.math) as typeof x;
    for (let i = 0; i < a.size; ++i) {
      const loc = a.indexToLoc(i);
      // Reverse location.
      const newLoc: number[] = new Array(loc.length);
      for (let j = 0; j < newLoc.length; j++) {
        newLoc[j] = dims_[j] ? a.shape[j] - loc[j] - 1 : loc[j];
      }

      const newIndex = result.locToIndex(newLoc);
      resultValues[newIndex] = values[i];
    }

    return result;
  }

  argmax(x: TensorDL, axis: number): TensorDL {
    ENV.setMath(x.math);
    return x.math.argMax(x, axis);
  }

  argmin(x: TensorDL, axis: number): TensorDL {
    ENV.setMath(x.math);
    return x.math.argMin(x, axis);
  }

  reduceSum(x: TensorDL, axes: number[], keepDims: boolean): TensorDL
  {
    ENV.setMath(x.math);
    return x.math.sum(x, axes, keepDims);
  }

  reduceMean(x: TensorDL, axes: number[], keepDims: boolean): TensorDL
  {
    ENV.setMath(x.math);
    return x.math.mean(x, axes, keepDims);
  }

  reduceMax(x: TensorDL, axes: number[], keepDims: boolean): TensorDL
  {
    ENV.setMath(x.math);
    return x.math.max(x, axes, keepDims);
  }

  equal(x: TensorDL, y: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.equal(x, y);
  }

  greater(x: TensorDL, y: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.greater(x, y);
  }

  greaterEqual(x: TensorDL, y: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.greaterEqual(x, y);
  }

  less(x: TensorDL, y: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.less(x, y);
  }

  lessEqual(x: TensorDL, y: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.lessEqual(x, y);
  }

  select(cond: TensorDL, t: TensorDL, f: TensorDL): TensorDL {
    const math = t.math;
    ENV.setMath(math);
    const condArray = math.cast(cond, "bool");
    return math.select(condArray, t, f);
  }

  sign(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const m = x.math;
    const a = m.step(x);  // maps neg to 0 and pos to 1
    // The following just does (2 * a - 1) which gives us sign.
    const dt = dtypeDL(x.dtype);
    const s2 = Scalar.new(2, dt);
    const s1 = Scalar.new(1, dt);
    const a2 = m.scalarTimesArray(s2, a);
    return m.arrayMinusScalar(a2, s1);
  }

  slice(x: TensorDL, begin: number[], size: number[]): TensorDL {
    ENV.setMath(x.math);
    let nd;
    // DL doesn't handle negative sizes, so we translate them.
    size = size.map((d, i) => {
      if (d >= 0) {
        return d;
      } else {
        assert(d === -1, "Bad value in size");
        return x.shape[i] - begin[i];
      }
    });
    switch (x.shape.length) {
      case 0:
        throw new Error("Slicing a scalar.");
      case 1:
        nd = x.math.slice1D(x.as1D(), begin[0], size[0]);
        break;
      case 2:
        nd = x.math.slice2D(x as Array2D,
                            [begin[0], begin[1]],
                            [size[0], size[1]]);
        break;
      case 3:
        nd = x.math.slice3D(x as Array3D,
                            [begin[0], begin[1], begin[2]],
                            [size[0], size[1], size[2]]);
        break;
      case 4:
        nd = x.math.slice4D(x as Array4D,
                            [begin[0], begin[1], begin[2], begin[3]],
                            [size[0], size[1], size[2], size[3]]);
        break;
      default:
        throw new Error("Slicing for tensors rank higher than " +
                        "4 not yet supported.");
    }
    return nd;
  }

  gather(x: TensorDL, indices: TensorDL, axis: number): TensorDL {
    ENV.setMath(x.math);
    return x.math.gather(x, indices as Array1D<"int32">, axis);
  }

  concat(axis: number, inputs: TensorDL[]): TensorDL {
    const m = inputs[0].math;
    ENV.setMath(m);
    const ndarrays = inputs;
    const shapes = inputs.map(t => t.shape);
    const rank = shapes[0].length;
    return ndarrays.reduce((a, b) => {
      if (rank === 0) {
        return m.concat1D(a.as1D(), b.as1D());
      } else if (rank === 1) {
        return m.concat1D(a.as1D(), b.as1D());
      } else if (rank === 2) {
        return m.concat2D(a as Array2D, b as Array2D, axis);
      } else if (rank === 3) {
        return m.concat3D(a as Array3D, b as Array3D, axis);
      } else if (rank === 4) {
        return m.concat4D(a as Array4D, b as Array4D, axis);
      } else {
        throw Error("Unsupported Tensor rank.");
      }
    });
  }

  pad(x: TensorDL, paddings: Array<[number, number]>,
      padValue: number): TensorDL {
    ENV.setMath(x.math);
    return x.math.pad(x, paddings, padValue);
  }

  reshape(x: TensorDL, newShape: types.Shape): TensorDL {
    ENV.setMath(x.math);
    return x.reshape(newShape);
  }

  softmax(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    return x.math.softmax(x);
  }

  logSoftmax(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const lastDim = x.shape.length - 1;
    return x.math.sub(x, x.math.logSumExp(x, lastDim, true));
  }

  cast(x: TensorDL, dtype: types.DType): TensorDL {
    ENV.setMath(x.math);
    return x.math.cast(x, dtypeDL(dtype));
  }

  oneHot(x: TensorDL, depth: number, onValue: number,
         offValue: number): TensorDL {
    ENV.setMath(x.math);
    const labels = x.math.cast(x, "float32").as1D();
    return x.math.oneHot(labels, depth, onValue, offValue);
  }

  conv2d(input: TensorDL, filter: TensorDL, opts: types.ConvOpts): TensorDL {
    ENV.setMath(input.math);
    return input.math.conv2d(input as Array4D, filter as Array4D,
                             null, opts.stride, opts.padding);
  }

  conv2dGradFilter(grad: TensorDL, input: TensorDL,
                   filterShape: types.Shape,
                   opts: types.ConvOpts): TensorDL {
    const math = input.math;
    ENV.setMath(math);
    return math.conv2dDerFilter(
      input as Array4D,
      grad as Array4D,
      filterShape as [number, number, number, number],
      opts.stride,
      opts.padding);
  }

  conv2dGradInput(grad: TensorDL, inputShape: types.Shape,
                  filter: TensorDL, opts: types.ConvOpts): TensorDL {
    const math = filter.math;
    ENV.setMath(math);
    return math.conv2dDerInput(
      inputShape as [number, number, number, number],
      grad as Array4D,
      filter as Array4D,
      opts.stride,
      opts.padding);
  }

  maxPool(input: TensorDL, opts: types.PoolOpts): TensorDL {
    ENV.setMath(input.math);
    return input.math.maxPool(input as Array4D, opts.size, opts.stride,
                              opts.padding);
  }

  maxPoolGrad(grad: TensorDL, origInput: TensorDL, origOutput: TensorDL,
              opts: types.PoolOpts): TensorDL {
    const m = grad.math;
    ENV.setMath(m);
    return m.maxPoolBackprop(grad, origInput, opts.size, opts.stride,
                             opts.padding);
  }
}
