// Deep Learn JS backend.
import { MatrixOrientation }
  from "./deps/deeplearnjs/src/math/backends/backend";
import { NDArrayMathCPU }
  from "./deps/deeplearnjs/src/math/backends/backend_cpu";
import { NDArrayMath } from "./deps/deeplearnjs/src/math/math";
import { Array2D, NDArray } from "./deps/deeplearnjs/src/math/ndarray";
import * as types from "./types";
import { assert } from "./util";

const cpuMath: NDArrayMathCPU = new NDArrayMathCPU();

export class BasicTensorDL implements types.BasicTensor {
  readonly dtype: types.DType;
  readonly shape: types.Shape;
  readonly math: NDArrayMath;
  readonly ndarray: NDArray;

  static fromTypedArray(data: types.TypedArray, shape: types.Shape):
    BasicTensorDL {
    const ndarray = NDArray.make(shape, { values: data });
    return new BasicTensorDL(ndarray, cpuMath);
  }

  constructor(ndarray: NDArray, math: NDArrayMath = cpuMath) {
    this.dtype = ndarray.dtype;
    this.shape = ndarray.shape;
    this.math = math;
    this.ndarray = ndarray;
  }

  getData(): types.TypedArray {
    return this.ndarray.getValues();
  }
}

export class BasicOpsDL implements types.BasicOps {
  add(x: BasicTensorDL, y: BasicTensorDL): BasicTensorDL {
    const ndarray = x.math.add(x.ndarray, y.ndarray);
    return new BasicTensorDL(ndarray, x.math);
  }

  sub(x: BasicTensorDL, y: BasicTensorDL): BasicTensorDL {
    const ndarray = x.math.sub(x.ndarray, y.ndarray);
    return new BasicTensorDL(ndarray, x.math);
  }

  mul(x: BasicTensorDL, y: BasicTensorDL): BasicTensorDL {
    const ndarray = x.math.multiply(x.ndarray, y.ndarray);
    return new BasicTensorDL(ndarray, x.math);
  }

  div(x: BasicTensorDL, y: BasicTensorDL): BasicTensorDL {
    const ndarray = x.math.divide(x.ndarray, y.ndarray);
    return new BasicTensorDL(ndarray, x.math);
  }

  neg(x: BasicTensorDL): BasicTensorDL {
    const ndarray = x.math.neg(x.ndarray);
    return new BasicTensorDL(ndarray, x.math);
  }

  exp(x: BasicTensorDL): BasicTensorDL {
    const ndarray = x.math.exp(x.ndarray);
    return new BasicTensorDL(ndarray, x.math);
  }

  log(x: BasicTensorDL): BasicTensorDL {
    const ndarray = x.math.log(x.ndarray);
    return new BasicTensorDL(ndarray, x.math);
  }

  matmul(x: BasicTensorDL, y: BasicTensorDL, transposeA = false,
         transposeB = false): BasicTensorDL {
    const f = (t) => t ?
      MatrixOrientation.TRANSPOSED :
      MatrixOrientation.REGULAR;
    assert(x.shape.length === 2 && y.shape.length === 2);
    const x2 = x.ndarray as Array2D;
    const y2 = y.ndarray as Array2D;
    const ndarray = x.math.matMul(x2, y2, f(transposeA), f(transposeB));
    return new BasicTensorDL(ndarray, x.math);
  }

  eye(size: number, dtype: types.DType = "float32"): types.BasicTensor {
    throw new Error("Not Implemented");
  }

  onesLike(x: BasicTensorDL): BasicTensorDL {
    const ones = NDArray.zerosLike(x.ndarray);
    ones.fill(1.0);
    return new BasicTensorDL(ones, x.math);
  }

  zerosLike(x: BasicTensorDL): BasicTensorDL {
    const zeros = NDArray.zerosLike(x.ndarray);
    return new BasicTensorDL(zeros, x.math);
  }

  square(x: BasicTensorDL): BasicTensorDL {
    const ndarray = x.math.square(x.ndarray);
    return new BasicTensorDL(ndarray, x.math);
  }

  sinh(x: BasicTensorDL): BasicTensorDL {
    const ndarray = x.math.sinh(x.ndarray);
    return new BasicTensorDL(ndarray, x.math);
  }

  cosh(x: BasicTensorDL): BasicTensorDL {
    const ndarray = x.math.cosh(x.ndarray);
    return new BasicTensorDL(ndarray, x.math);
  }

  tanh(x: BasicTensorDL): BasicTensorDL {
    const ndarray = x.math.tanh(x.ndarray);
    return new BasicTensorDL(ndarray, x.math);
  }

  randn(shape: types.Shape, seed?: number): BasicTensorDL {
    const r = NDArray.randNormal(shape, 0, 1, "float32", seed);
    return new BasicTensorDL(r, cpuMath);
  }

  linspace(start: number, stop: number, num: number): BasicTensorDL {
    const d = (stop - start) / (num - 1);
    const ta = new Float32Array(num);
    for (let i = 0; i <= num - 1; ++i) {
      ta[i] = start + i * d;
    }
    return BasicTensorDL.fromTypedArray(ta, [num]);
  }

  arange(start: number, limit: number, delta: number): BasicTensorDL {
    const num = (limit - start) / delta;
    const ta = new Int32Array(num);
    for (let i = 0; i < num; ++i) {
      ta[i] = start + i * delta;
    }
    return BasicTensorDL.fromTypedArray(ta, [num]);
  }

  transpose(x: BasicTensorDL, perm: BasicTensorDL): BasicTensorDL {
    const permArr = Array.from(perm.ndarray.getValues());
    const ndarray = x.math.transpose(x.ndarray, permArr);
    return new BasicTensorDL(ndarray, x.math);
  }

  reverse(x: BasicTensorDL, dims: BasicTensorDL): BasicTensorDL {
    const a = x.ndarray;
    const dims_ = dims.getData();
    // TODO move to deeplearnjs/src/math/backends/backend_cpu.ts
    const resultValues = types.makeTypedArray(a.size, x.dtype);
    const values = a.getValues();
    const result = NDArray.make(a.shape, {values: resultValues});
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

    return new BasicTensorDL(result, x.math);
  }

  reduceSum(x: BasicTensorDL, axes: number[], keepDims: boolean): BasicTensorDL
  {
    const ndarray = x.math.sum(x.ndarray, axes, keepDims);
    return new BasicTensorDL(ndarray, x.math);
  }

  reduceMax(x: BasicTensorDL, axes: number[], keepDims: boolean): BasicTensorDL
  {
    const ndarray = x.math.max(x.ndarray, axes, keepDims);
    return new BasicTensorDL(ndarray, x.math);
  }

  equal(x: BasicTensorDL, y: BasicTensorDL): BasicTensorDL {
    const ndarray = x.math.equal(x.ndarray, y.ndarray);
    return new BasicTensorDL(ndarray, x.math);
  }

  reshape(x: BasicTensorDL, newShape: types.Shape): BasicTensorDL {
    const ndarray = x.ndarray.reshape(newShape);
    return new BasicTensorDL(ndarray, x.math);
  }

  softmax(x: BasicTensorDL): BasicTensorDL {
    const ndarray = x.math.softmax(x.ndarray);
    return new BasicTensorDL(ndarray, x.math);
  }

  logSoftmax(x: BasicTensorDL): BasicTensorDL {
    const xa = x.ndarray;
    const lastDim = x.shape.length - 1;
    const ndarray = x.math.sub(xa, x.math.logSumExp(xa, lastDim, true));
    return new BasicTensorDL(ndarray, x.math);
  }
}
