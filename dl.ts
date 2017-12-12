// Deep Learn JS backend.
import { MatrixOrientation }
  from "./deps/deeplearnjs/src/math/backends/backend";
import { NDArrayMathCPU }
  from "./deps/deeplearnjs/src/math/backends/backend_cpu";
import { NDArrayMath } from "./deps/deeplearnjs/src/math/math";
import { Array2D, Array3D, Array4D, NDArray }
  from "./deps/deeplearnjs/src/math/ndarray";
import * as types from "./types";
import { assert } from "./util";

const cpuMath: NDArrayMathCPU = new NDArrayMathCPU();

export class BasicTensorDL implements types.BasicTensor {
  readonly dtype: types.DType;
  readonly shape: types.Shape;
  readonly math: NDArrayMath;
  readonly ndarray: NDArray;

  static fromTypedArray(data: types.TypedArray, shape: types.Shape,
                        dtype?: types.DType): BasicTensorDL {
    if (dtype === undefined) {
      dtype = types.getDType(data);
    }
    const ndarray = NDArray.make(shape, { values: data }, dtype as any);
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

  fill(value: BasicTensorDL, shape: types.Shape): BasicTensorDL {
    if (value.shape.length !== 0) {
      throw new Error("Fill value must be a scalar.");
    }
    const out = NDArray.zeros(shape, value.ndarray.dtype);
    out.fill(value.ndarray.getValues()[0]);
    return new BasicTensorDL(out, value.math);
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

  argmax(x: BasicTensorDL, axis: number): BasicTensorDL {
    const ndarray = x.math.argMax(x.ndarray, axis);
    return new BasicTensorDL(ndarray, x.math);
  }

  argmin(x: BasicTensorDL, axis: number): BasicTensorDL {
    const ndarray = x.math.argMin(x.ndarray, axis);
    return new BasicTensorDL(ndarray, x.math);
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

  slice(x: BasicTensorDL, begin: number[], size: number[]): BasicTensorDL {
    let nd;
    // DL doesn't handle negative sizes, so we translate them.
    size = size.map((d, i) => {
      if (d >= 0) {
        return d;
      } else {
        assert(d === -1, "Bad value in size");
        return x.shape[i];
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
    return new BasicTensorDL(nd, x.math);
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

  cast(x: BasicTensorDL, dtype: types.DType): BasicTensorDL {
    const nd = NDArray.make(x.shape, { values: x.getData() }, dtype as any);
    return new BasicTensorDL(nd, x.math);
  }

  oneHot(x: BasicTensorDL, depth: number, onValue: number,
         offValue: number): BasicTensorDL {
    const nd = x.math.oneHot(x.ndarray.as1D(), depth, onValue, offValue);
    return new BasicTensorDL(nd, x.math);
  }
}
