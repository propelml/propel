// Deep Learn JS backend.
import * as types from "./types";
import { NDArray } from "./deps/deeplearnjs/src/math/ndarray";
import { NDArrayMath } from "./deps/deeplearnjs/src/math/math";
import {NDArrayMathCPU} from './deps/deeplearnjs/src/math/backends/backend_cpu';

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
}
