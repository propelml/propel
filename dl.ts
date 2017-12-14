// Deep Learn JS backend.
import { NDArrayMathCPU }
  from "./deps/deeplearnjs/src/math/backends/backend_cpu";
import { MatrixOrientation }
  from "./deps/deeplearnjs/src/math/backends/types/matmul";
import { NDArrayMath } from "./deps/deeplearnjs/src/math/math";
import { Array2D, Array3D, Array4D, NDArray }
  from "./deps/deeplearnjs/src/math/ndarray";
import * as types from "./types";
import { assert } from "./util";

const cpuMath: NDArrayMathCPU = new NDArrayMathCPU();

export class TensorDL implements types.BasicTensor {
  readonly dtype: types.DType;
  readonly shape: types.Shape;
  readonly math: NDArrayMath;
  readonly ndarray: NDArray;

  static fromTypedArray(data: types.TypedArray, shape: types.Shape,
                        dtype?: types.DType): TensorDL {
    if (dtype === undefined) {
      dtype = types.getDType(data);
    }
    const ndarray = NDArray.make(shape, { values: data }, dtype as any);
    return new TensorDL(ndarray, cpuMath);
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

export class OpsDL implements types.BackendOps {
  add(x: TensorDL, y: TensorDL): TensorDL {
    const ndarray = x.math.add(x.ndarray, y.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  sub(x: TensorDL, y: TensorDL): TensorDL {
    const ndarray = x.math.sub(x.ndarray, y.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  mul(x: TensorDL, y: TensorDL): TensorDL {
    const ndarray = x.math.multiply(x.ndarray, y.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  div(x: TensorDL, y: TensorDL): TensorDL {
    const ndarray = x.math.divide(x.ndarray, y.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  neg(x: TensorDL): TensorDL {
    const ndarray = x.math.neg(x.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  exp(x: TensorDL): TensorDL {
    const ndarray = x.math.exp(x.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  log(x: TensorDL): TensorDL {
    const ndarray = x.math.log(x.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  matmul(x: TensorDL, y: TensorDL, transposeA = false,
         transposeB = false): TensorDL {
    const f = (t) => t ?
      MatrixOrientation.TRANSPOSED :
      MatrixOrientation.REGULAR;
    assert(x.shape.length === 2 && y.shape.length === 2);
    const x2 = x.ndarray as Array2D;
    const y2 = y.ndarray as Array2D;
    const ndarray = x.math.matMul(x2, y2, f(transposeA), f(transposeB));
    return new TensorDL(ndarray, x.math);
  }

  eye(size: number, dtype: types.DType = "float32"): types.BasicTensor {
    throw new Error("Not Implemented");
  }

  onesLike(x: TensorDL): TensorDL {
    const ones = NDArray.zerosLike(x.ndarray);
    ones.fill(1.0);
    return new TensorDL(ones, x.math);
  }

  zerosLike(x: TensorDL): TensorDL {
    const zeros = NDArray.zerosLike(x.ndarray);
    return new TensorDL(zeros, x.math);
  }

  fill(value: TensorDL, shape: types.Shape): TensorDL {
    if (value.shape.length !== 0) {
      throw new Error("Fill value must be a scalar.");
    }
    const out = NDArray.zeros(shape, value.ndarray.dtype);
    out.fill(value.ndarray.getValues()[0]);
    return new TensorDL(out, value.math);
  }

  square(x: TensorDL): TensorDL {
    const ndarray = x.math.square(x.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  sinh(x: TensorDL): TensorDL {
    const ndarray = x.math.sinh(x.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  cosh(x: TensorDL): TensorDL {
    const ndarray = x.math.cosh(x.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  tanh(x: TensorDL): TensorDL {
    const ndarray = x.math.tanh(x.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  randn(shape: types.Shape, seed?: number): TensorDL {
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

  arange(start: number, limit: number, delta: number): TensorDL {
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

  reverse(x: TensorDL, dims: TensorDL): TensorDL {
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

    return new TensorDL(result, x.math);
  }

  argmax(x: TensorDL, axis: number): TensorDL {
    const ndarray = x.math.argMax(x.ndarray, axis);
    return new TensorDL(ndarray, x.math);
  }

  argmin(x: TensorDL, axis: number): TensorDL {
    const ndarray = x.math.argMin(x.ndarray, axis);
    return new TensorDL(ndarray, x.math);
  }

  reduceSum(x: TensorDL, axes: number[], keepDims: boolean): TensorDL
  {
    const ndarray = x.math.sum(x.ndarray, axes, keepDims);
    return new TensorDL(ndarray, x.math);
  }

  reduceMean(x: TensorDL, axes: number[], keepDims: boolean): TensorDL
  {
    const ndarray = x.math.mean(x.ndarray, axes, keepDims);
    return new TensorDL(ndarray, x.math);
  }

  reduceMax(x: TensorDL, axes: number[], keepDims: boolean): TensorDL
  {
    const ndarray = x.math.max(x.ndarray, axes, keepDims);
    return new TensorDL(ndarray, x.math);
  }

  equal(x: TensorDL, y: TensorDL): TensorDL {
    const ndarray = x.math.equal(x.ndarray, y.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  slice(x: TensorDL, begin: number[], size: number[]): TensorDL {
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
    return new TensorDL(nd, x.math);
  }

  reshape(x: TensorDL, newShape: types.Shape): TensorDL {
    const ndarray = x.ndarray.reshape(newShape);
    return new TensorDL(ndarray, x.math);
  }

  softmax(x: TensorDL): TensorDL {
    const ndarray = x.math.softmax(x.ndarray);
    return new TensorDL(ndarray, x.math);
  }

  logSoftmax(x: TensorDL): TensorDL {
    const xa = x.ndarray;
    const lastDim = x.shape.length - 1;
    const ndarray = x.math.sub(xa, x.math.logSumExp(xa, lastDim, true));
    return new TensorDL(ndarray, x.math);
  }

  cast(x: TensorDL, dtype: types.DType): TensorDL {
    const nd = NDArray.make(x.shape, { values: x.getData() }, dtype as any);
    return new TensorDL(nd, x.math);
  }

  oneHot(x: TensorDL, depth: number, onValue: number,
         offValue: number): TensorDL {
    const nd = x.math.oneHot(x.ndarray.as1D(), depth, onValue, offValue);
    return new TensorDL(nd, x.math);
  }
}
