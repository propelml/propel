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
import * as types from "./types";
import { assert, getDType, makeTypedArray } from "./util";

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

export class TensorDL implements types.BasicTensor {
  readonly dtype: types.DType;
  readonly shape: types.Shape;
  // We are aware that ndarray already has a math object on it.  We want more
  // explicit control over how DL operates, plus ndarray.math is private. So we
  // just make an extra reference to it here.
  readonly math: NDArrayMath;
  readonly ndarray: NDArray;
  private isDisposed: boolean;

  static fromTypedArray(data: types.TypedArray, shape: types.Shape,
                        dtype?: types.DType, device?: string): TensorDL {
    if (dtype == null) {
      dtype = getDType(data);
    }
    if (device == null) {
      device = "CPU:0";
    }
    const math = lookupMath(device);
    const ndarray = NDArray.make(shape, { values: data }, dtype as any, math);
    return new TensorDL(ndarray, math);
  }

  constructor(ndarray: NDArray, math: NDArrayMath = cpuMath) {
    this.dtype = ndarray.dtype;
    this.shape = ndarray.shape;
    this.math = math;
    this.ndarray = ndarray;
    this.isDisposed = false;
    assert((this.ndarray as any).isDisposed === false);
  }

  dataSync(): types.TypedArray {
    assert(!this.isDisposed);
    return this.ndarray.getValues();
  }

  data(): Promise<types.TypedArray> {
    assert(!this.isDisposed);
    return this.ndarray.data();
  }

  dispose(): void {
    // Currently this asserts that TensorDL should only have dispose() called
    // once on it. However, there may be legitimate situations where calling
    // dispose() twice could occur. Leaving it for now.
    assert(!this.isDisposed);
    if (!(this.ndarray as any).isDisposed) {
      this.ndarray.dispose();
    }
    this.isDisposed = true;
  }
}

export class OpsDL implements types.BackendOps {

  copyToDevice(x: TensorDL, device: string): TensorDL {
    const math = lookupMath(device);
    const orig = x.ndarray;
    // TODO orig.dataSync() is synchronous
    const nd = NDArray.make(orig.shape, {values: orig.dataSync()},
                            orig.dtype, math);
    return new TensorDL(nd, math);

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

  add(x: TensorDL, y: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.add(x.ndarray, y.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  sub(x: TensorDL, y: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.sub(x.ndarray, y.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  mul(x: TensorDL, y: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.multiply(x.ndarray, y.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  div(x: TensorDL, y: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.divide(x.ndarray, y.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  neg(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.neg(x.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  exp(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.exp(x.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  log(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.log(x.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  matmul(x: TensorDL, y: TensorDL, transposeA = false,
         transposeB = false): TensorDL {
    ENV.setMath(x.math);
    const f = (t) => t ?
      MatrixOrientation.TRANSPOSED :
      MatrixOrientation.REGULAR;
    assert(x.shape.length === 2 && y.shape.length === 2);
    const x2 = x.ndarray as Array2D;
    const y2 = y.ndarray as Array2D;
    const ndarray = x.math.matMul(x2, y2, f(transposeA), f(transposeB));
    return new TensorDL(ndarray, x.math);
  }

  setDiag(x: TensorDL, diag: TensorDL): TensorDL {
    if (x.shape.length !== 2 || diag.shape.length !== 1) {
      throw new Error("Not implemented");
    }
    // DL doesn't support WebGL for this yet, so force CPU.
    ENV.setMath(cpuMath);
    const nd = cpuMath.setDiag(x.ndarray as Array2D, diag.ndarray as Array1D);
    return new TensorDL(nd, x.math);
  }

  onesLike(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ones = NDArray.zerosLike(x.ndarray);
    ones.fill(1.0);
    return new TensorDL(ones, x.math);
  }

  zerosLike(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const zeros = NDArray.zerosLike(x.ndarray);
    return new TensorDL(zeros, x.math);
  }

  fill(value: TensorDL, shape: types.Shape): TensorDL {
    if (value.shape.length !== 0) {
      throw new Error("Fill value must be a scalar.");
    }
    ENV.setMath(value.math);
    const out = NDArray.zeros(shape, value.ndarray.dtype);
    out.fill(value.ndarray.getValues()[0]);
    return new TensorDL(out, value.math);
  }

  square(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.square(x.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  sinh(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.sinh(x.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  cosh(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.cosh(x.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  tanh(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.tanh(x.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  relu(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.relu(x.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  reluGrad(grad: TensorDL, features: TensorDL): TensorDL {
    const m = grad.math;
    const s = m.step(features.ndarray);
    const ndarray = m.multiply(grad.ndarray, s);
    return new TensorDL(ndarray, m);
  }

  sigmoid(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.sigmoid(x.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  abs(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.abs(x.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  randn(shape: types.Shape, seed?: number): TensorDL {
    ENV.setMath(cpuMath);
    const r = NDArray.randNormal(shape, 0, 1, "float32", seed);
    return new TensorDL(r, cpuMath);
  }

  linspace(start: number, stop: number, num: number): TensorDL {
    const d = (stop - start) / (num - 1);
    const ta = new Float32Array(num);
    for (let i = 0; i <= num - 1; ++i) {
      ta[i] = start + i * d;
    }
    return TensorDL.fromTypedArray(ta, [num]);
  }

  range(start: number, limit: number, delta: number): TensorDL {
    const num = (limit - start) / delta;
    const ta = new Int32Array(num);
    for (let i = 0; i < num; ++i) {
      ta[i] = start + i * delta;
    }
    return TensorDL.fromTypedArray(ta, [num]);
  }

  transpose(x: TensorDL, perm: TensorDL): TensorDL {
    const permArr = Array.from(perm.ndarray.getValues());
    const ndarray = x.math.transpose(x.ndarray, permArr);
    return new TensorDL(ndarray, x.math);
  }

  // TODO dims should not be a tensor.
  reverse(x: TensorDL, dims: TensorDL): TensorDL {
    const a = x.ndarray;
    const dims_ = dims.dataSync();
    // TODO move to deeplearnjs/src/math/backends/backend_cpu.ts
    const resultValues = makeTypedArray(a.size, x.dtype);
    const values = a.getValues();
    const dtype = dtypeDL(x.dtype);
    const result = NDArray.make(a.shape, {values: resultValues},
                                dtype) as typeof x.ndarray;
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

    return new TensorDL(result, x.math);
  }

  argmax(x: TensorDL, axis: number): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.argMax(x.ndarray, axis);
    return new TensorDL(ndarray, x.math);
  }

  argmin(x: TensorDL, axis: number): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.argMin(x.ndarray, axis);
    return new TensorDL(ndarray, x.math);
  }

  reduceSum(x: TensorDL, axes: number[], keepDims: boolean): TensorDL
  {
    ENV.setMath(x.math);
    const ndarray = x.math.sum(x.ndarray, axes, keepDims);
    return new TensorDL(ndarray, x.math);
  }

  reduceMean(x: TensorDL, axes: number[], keepDims: boolean): TensorDL
  {
    ENV.setMath(x.math);
    const ndarray = x.math.mean(x.ndarray, axes, keepDims);
    return new TensorDL(ndarray, x.math);
  }

  reduceMax(x: TensorDL, axes: number[], keepDims: boolean): TensorDL
  {
    ENV.setMath(x.math);
    const ndarray = x.math.max(x.ndarray, axes, keepDims);
    return new TensorDL(ndarray, x.math);
  }

  equal(x: TensorDL, y: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.equal(x.ndarray, y.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  greater(x: TensorDL, y: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.greater(x.ndarray, y.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  greaterEqual(x: TensorDL, y: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.greaterEqual(x.ndarray, y.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  less(x: TensorDL, y: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.less(x.ndarray, y.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  lessEqual(x: TensorDL, y: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.lessEqual(x.ndarray, y.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  select(cond: TensorDL, t: TensorDL, f: TensorDL): TensorDL {
    const math = t.math;
    ENV.setMath(math);
    const condArray = math.cast(cond.ndarray, "bool");
    const ndarray = math.select(condArray, t.ndarray, f.ndarray);
    return new TensorDL(ndarray, math);
  }

  sign(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const m = x.math;
    const a = m.step(x.ndarray);  // maps neg to 0 and pos to 1
    // The following just does (2 * a - 1) which gives us sign.
    const dt = dtypeDL(x.dtype);
    const s2 = Scalar.new(2, dt);
    const s1 = Scalar.new(1, dt);
    const a2 = m.scalarTimesArray(s2, a);
    const b = m.arrayMinusScalar(a2, s1);
    return new TensorDL(b, m);
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
        nd = x.math.slice1D(x.ndarray.as1D(), begin[0], size[0]);
        break;
      case 2:
        nd = x.math.slice2D(x.ndarray as Array2D,
                            [begin[0], begin[1]],
                            [size[0], size[1]]);
        break;
      case 3:
        nd = x.math.slice3D(x.ndarray as Array3D,
                            [begin[0], begin[1], begin[2]],
                            [size[0], size[1], size[2]]);
        break;
      case 4:
        nd = x.math.slice4D(x.ndarray as Array4D,
                            [begin[0], begin[1], begin[2], begin[3]],
                            [size[0], size[1], size[2], size[3]]);
        break;
      default:
        throw new Error("Slicing for tensors rank higher than " +
                        "4 not yet supported.");
    }
    // DL has a bug. No matter what the dtype being sliced, it always returns a
    // float32. The proper thing would be to fix the bug inside DL, but we're
    // going to hack it for now.
    nd = nd.asType(x.dtype);
    return new TensorDL(nd, x.math);
  }

  concat(axis: number, inputs: TensorDL[]): TensorDL {
    const m = inputs[0].math;
    ENV.setMath(m);
    const ndarrays = inputs.map(t => t.ndarray);
    const shapes = inputs.map(t => t.shape);
    const rank = shapes[0].length;
    const nd = ndarrays.reduce((a, b) => {
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
    return new TensorDL(nd, m);
  }

  reshape(x: TensorDL, newShape: types.Shape): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.ndarray.reshape(newShape);
    return new TensorDL(ndarray, x.math);
  }

  softmax(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const ndarray = x.math.softmax(x.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  logSoftmax(x: TensorDL): TensorDL {
    ENV.setMath(x.math);
    const xa = x.ndarray;
    const lastDim = x.shape.length - 1;
    const ndarray = x.math.sub(xa, x.math.logSumExp(xa, lastDim, true));
    return new TensorDL(ndarray, x.math);
  }

  cast(x: TensorDL, dtype: types.DType): TensorDL {
    ENV.setMath(x.math);
    const nd = x.math.cast(x.ndarray, dtypeDL(dtype));
    return new TensorDL(nd, x.math);
  }

  oneHot(x: TensorDL, depth: number, onValue: number,
         offValue: number): TensorDL {
    ENV.setMath(x.math);
    const labels = x.math.cast(x.ndarray, "float32").as1D();
    const nd = x.math.oneHot(labels, depth, onValue, offValue);
    return new TensorDL(nd, x.math);
  }
}
