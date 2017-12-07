// Deep Learn JS backend.
import { NDArrayMathCPU }
  from "./deps/deeplearnjs/src/math/backends/backend_cpu";
import { NDArrayMath } from "./deps/deeplearnjs/src/math/math";
import { NDArray } from "./deps/deeplearnjs/src/math/ndarray";
import * as types from "./types";

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

  eye(size: number, dtype: types.DType = "float32"): types.BasicTensor {
    throw new Error("Not Implemented");
  }

  onesLike(x: BasicTensorDL): BasicTensorDL {
    const ones = NDArray.zerosLike(x.ndarray);
    ones.fill(1.0);
    return new BasicTensorDL(ones, x.math);
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
}
