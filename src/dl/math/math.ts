/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import { BackendType, ENV } from "../environment";
import * as util from "../util";
import * as axis_util from "./axis_util";
import { MathBackend } from "./backends/backend";
import { BackendEngine, ScopeResult } from "./backends/backend_engine";
import * as broadcast_util from "./broadcast_util";
import * as concat_util from "./concat_util";
import * as conv_util from "./conv_util";
import { Array1D, Array2D, Array3D, Array4D, DataId, DataType, DataTypeMap,
  IntDType, NDArray, Rank, RankMap, Scalar } from "./ndarray";
import * as slice_util from "./slice_util";
import { MatrixOrientation, SumTypes } from "./types";

export interface LSTMCell {
  (data: Array2D, c: Array2D, h: Array2D): [Array2D, Array2D];
}

export interface NDArrayManager {
  getNumArrays(): number;
  register(a: NDArray): void;
}

export class NDArrayMath implements NDArrayManager {
  protected backendEngine: BackendEngine;
  private registeredArrays = new WeakMap<DataId, number>();
  private registeredArrayCount = 0;
  private backend: MathBackend;
  private customBackend = false;

  time(query: () => NDArray): Promise<number> {
    return this.backend.time(query);
  }

  getNumArrays() {
    return this.registeredArrayCount;
  }

  register(a: NDArray): void {
    const refCount = this.registeredArrays.has(a.dataId) ?
        this.registeredArrays.get(a.dataId) :
        0;
    if (refCount === 0) {
      this.backend.register(a.dataId, a.shape, a.dtype);
      this.registeredArrayCount++;
    }
    this.registeredArrays.set(a.dataId, refCount + 1);
    this.backendEngine.track(a);
  }

  writePixels(
      dataId: DataId,
      pixels: ImageData | HTMLImageElement | HTMLCanvasElement |
              HTMLVideoElement,
      numChannels: number): void {
    this.backend.writePixels(dataId, pixels, numChannels);
  }
  write<D extends DataType>(dataId: DataId, values: DataTypeMap[D]): void {
    this.backend.write(dataId, values);
  }
  readSync<D extends DataType>(dataId: DataId): DataTypeMap[D] {
    return this.backend.readSync(dataId);
  }
  read<D extends DataType>(dataId: DataId): Promise<DataTypeMap[D]> {
    return this.backend.read(dataId);
  }

  /**
   * @param safeMode In safe mode, you must use math operations inside
   *     a math.scope() which will automatically clean up intermediate NDArrays.
   */
  constructor(backend: BackendType | MathBackend) {
    if (typeof backend === "string") {
      this.backend = ENV.getBackend(backend);
    } else {
      this.customBackend = true;
      this.backend = backend;
    }
    this.backendEngine = new BackendEngine();
  }

  /**
   * Create a new math scope. Put chained math operations inside a scope
   * function closure so that the library automatically cleans up NDArrays
   * from intermediate math operations. You must create a scope in safe mode
   * to call math operations. If a result is returned from the scope, it will
   * also be tracked, which means there must be yet another wrapping scope.
   * @param scopeFn The function to execute with chained math operations.
   */
  scope<T extends ScopeResult>(scopeFn: () => T): T {
    return this.backendEngine.scope("scope", scopeFn);
  }

  /**
   * Start a scope. Use this with endScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  startScope() {
    this.backendEngine.startScope();
  }

  /**
   * End a scope. Use this with startScope() to achieve the same functionality
   * as scope() without the need for a function closure.
   */
  endScope(result: {}) {
    this.backendEngine.endScope(result);
  }

  dispose() {
    if (this.customBackend) {
      this.backend.dispose();
    }
  }

  /**
   * Computes the dot product of two matrices, A * B. These must be matrices,
   * use matrixTimesVector and vectorTimesMatrix, dotProduct, and outerProduct
   * in other cases.
   * @param a First matrix in dot product operation.
   * @param b Second matrix in dot product operation.
   * @param aOrientation The MatrixOrientation of A. If using TRANSPOSED, will
   * compute A^T * B.
   * @param bOrientation The MatrixOrientation of B. If using TRANSPOSED, will
   * compute A * B^T.
   */
  matMul(
      a: Array2D, b: Array2D, aOrientation = MatrixOrientation.REGULAR,
      bOrientation = MatrixOrientation.REGULAR): Array2D {
    const innerShapeA =
        (aOrientation === MatrixOrientation.REGULAR) ? a.shape[1] : a.shape[0];
    const innerShapeB =
        (bOrientation === MatrixOrientation.REGULAR) ? b.shape[0] : b.shape[1];

    util.assert(
        a.rank === 2 && b.rank === 2,
        `Error in matMul: inputs must be rank 2, got ranks ${a.rank}` +
            ` and ${b.rank}.`);

    util.assert(
        innerShapeA === innerShapeB,
        `Error in matMul: inner shapes (${innerShapeA}) and (` +
            `${innerShapeB}) of NDArrays with shapes ${a.shape} and ` +
            `${b.shape} and orientations ${MatrixOrientation[aOrientation]}` +
            ` and ${MatrixOrientation[bOrientation]} must match.`);

    return this.backend.matMul(a, b, aOrientation, bOrientation);
  }

  private executeOp<T extends NDArray>(name: string, f: () => T): T {
    // TODO(nsthorat): Do operation logging and performance profiling.
    return f();
  }

  /**
   * Computes the dot product of a vector and a matrix, v * B.
   * @param v The vector in dot product operation.
   * @param matrix The matrix in dot product operation.
   */
  vectorTimesMatrix(v: Array1D, matrix: Array2D): Array1D {
    util.assert(
        v.rank === 1,
        `Error in vectorTimesMatrix: first input must be rank 1, but got ` +
            `rank ${v.rank}.`);
    util.assert(
        matrix.rank === 2,
        `Error in vectorTimesMatrix: second input must be rank 2, but got ` +
            `rank ${matrix.rank}.`);
    util.assert(
        v.size === matrix.shape[0],
        `Error in vectorTimesMatrix: size of vector (${v.size}) ` +
            `must match first dimension of matrix (${matrix.shape[0]})`);

    return this.matMul(v.as2D(1, -1), matrix).as1D();
  }

  /**
   * Computes the dot product of a matrix and vector, A * v.
   * @param matrix The matrix in dot product operation.
   * @param v The vector in dot product operation.
   */
  matrixTimesVector(matrix: Array2D, v: Array1D): Array1D {
    util.assert(
        v.rank === 1,
        `Error in matrixTimesVector: second input must rank 1, but got ` +
            `rank ${v.rank}.`);
    util.assert(
        matrix.rank === 2,
        `Error in matrixTimesVector: first input must be a rank 2, but got ` +
            `rank ${matrix.rank}.`);
    util.assert(
        v.size === matrix.shape[1],
        `Error in matrixTimesVector: size of first rank 1 input ${v.size} ` +
            `must match inner dimension of second rank 2 input, but got ` +
            `shape ${matrix.shape}.`);

    return this.matMul(matrix, v.as2D(-1, 1)).as1D();
  }

  /**
   * Computes the dot product of two vectors, v1 * v2.
   * @param v1 The first vector in the dot product operation.
   * @param v2 The second vector in the dot product operation.
   */
  dotProduct(v1: Array1D, v2: Array1D): Scalar {
    util.assert(
        v1.rank === 1 && v2.rank === 1,
        `Error in dotProduct: inputs must be rank 1, but got ranks ` +
            `${v1.rank} and ${v2.rank}.`);
    util.assert(
        v1.size === v2.size,
        `Error in dotProduct: size of inputs (${v1.size}) and (` +
            `${v2.size}) must match.`);
    return this.matMul(v1.as2D(1, -1), v2.as2D(-1, 1)).asScalar();
  }

  /**
   * Computes the outer product of two vectors, v1 and v2.
   * @param v1 The first vector in the outer product operation.
   * @param v2 The second vector in the dot product operation.
   */
  outerProduct(v1: Array1D, v2: Array1D): Array2D {
    util.assert(
        v1.rank === 1 && v2.rank === 1,
        `Error in outerProduct: inputs must be rank 1, but got ranks ` +
            `${v1.rank} and ${v2.rank}.`);

    return this.matMul(v1.as2D(-1, 1), v2.as2D(1, -1));
  }

  ///////////////
  // Shape ops //
  ///////////////

  /**
   * Clones an NDArray of any shape.
   * @param x The NDArray to clone.
   */
  clone<T extends NDArray>(x: T): T {
    return this.backend.clone(x);
  }

  /** Reshapes the array. */
  reshape<D extends DataType, R extends Rank, T extends RankMap<D>[R]>(
      x: NDArray<D>, newShape: number[]): T {
    newShape = util.inferFromImplicitShape(newShape, x.size);
    util.assert(
        x.size === util.sizeFromShape(newShape),
        "new shape and old shape must have the same number of elements.");

    return NDArray.make(newShape, {dataId: x.dataId}, x.dtype) as T;
  }

  /**
   * Casts a tensor to a new type. If the new type matches the old type,
   * this is a no-op.
   */
  cast<D extends DataType, R extends Rank>(
      x: NDArray<DataType, R>, newDType: D): RankMap<D>[R] {
    if (!util.hasEncodingLoss(x.dtype, newDType)) {
      // We don't change the underlying data, since we cast to higher precision.
      return NDArray.make(x.shape, {dataId: x.dataId}, newDType);
    }
    if (newDType === "int32" || newDType === "uint8") {
      return this.backend.int(x, newDType as IntDType) as NDArray as
        RankMap<D>[R];
    } else if (newDType === "bool") {
      return this.backend.notEqual(x, Scalar.new(0, x.dtype)) as RankMap<D>[R];
    } else {
      throw new Error(`Error in Cast: unknown dtype argument (${newDType})`);
    }
  }

  gather(x: NDArray, indices: Array1D<"int32">, axis: number): NDArray {
    return this.backend.gather(x, indices, axis);
  }

  /**
   * Extracts a 1D slice from 1D array starting at coordinates `begin` and is
   * of length `size`.
   *
   * @param x The input array to slice from.
   * @param begin The offset to start the slice from.
   * @param size The size of the slice.
   */
  slice1D(x: Array1D, begin: number, size: number): Array1D {
    slice_util.assertParamsValid(x, [begin], [size]);
    return this.backend.slice1D(x, begin, size);
  }

  /**
   * Extracts a 2D slice from a 2D array starting at coordinates `begin` and
   * is of size `size`.
   *
   * @param x The input array to slice from.
   * @param begin The [row, col] 2d coordinates to start the slice from.
   * @param size The size of the slice.
   */
  slice2D(x: Array2D, begin: [number, number], size: [number, number]):
      Array2D {
    slice_util.assertParamsValid(x, begin, size);
    return this.backend.slice2D(x, begin, size);
  }

  /**
   * Extracts a 3D slice from a 3D array starting at coordinates `begin` and
   * is of size `size`.
   *
   * @param x The input array to slice from.
   * @param begin The [row, col, depth] 3d coordinates to start the slice from.
   * @param size The size of the slice.
   */
  slice3D(x: Array3D, begin: [number, number, number], size: [
    number, number, number
  ]): Array3D {
    slice_util.assertParamsValid(x, begin, size);
    return this.backend.slice3D(x, begin, size);
  }

  /**
   * Extracts a 4D slice from a 4D array starting at coordinates `begin` and
   * is of size `size`.
   *
   * @param x The input array to slice from.
   * @param begin The [row, col, depth, depth2] 4d coordinates to start the
   *              slice from.
   * @param size The size of the slice.
   */
  slice4D(x: Array4D, begin: [number, number, number, number], size: [
    number, number, number, number
  ]): Array4D {
    slice_util.assertParamsValid(x, begin, size);
    return this.backend.slice4D(x, begin, size);
  }

  /**
   * Concatenates two 1D arrays.
   *
   * For example, if:
   * A: shape(3) = |r1, g1, b1|
   * B: shape(2) = |r2, g2|
   * C = concat1D(A, B) == |r1, g1, b1, r2, g2|
   *
   * @param a The first array.
   * @param b The second array.
   * @return The concatenated array.
   */
  concat1D(a: Array1D, b: Array1D): Array1D {
    concat_util.assertParams(a.shape, b.shape, 0);
    return this.backend.concat1D(a, b);
  }

  /**
   * Concatenates two 2D arrays along a given axis.
   *
   * For example, if:
   * A: shape(2, 3) = | r1, g1, b1 |
   *                  | r2, g2, b2 |
   *
   * B: shape(2, 3) = | r3, g3, b3 |
   *                  | r4, g4, b4 |
   *
   * C = concat2D(A, B, axis)
   *
   * if axis = 0:
   * C: shape(4, 3) = | r1, g1, b1 |
   *                  | r2, g2, b2 |
   *                  | r3, g3, b3 |
   *                  | r4, g4, b4 |
   *
   * if axis = 1:
   * C = shape(2, 6) = | r1, g1, b1, r3, g3, b3 |
   *                   | r2, g2, b2, r4, g4, b4 |
   *
   *
   * @param a The first array.
   * @param b The second array.
   * @param axis The axis to concatenate along.
   * @return The concatenated array.
   */
  concat2D(a: Array2D, b: Array2D, axis: number): Array2D {
    concat_util.assertParams(a.shape, b.shape, axis);
    return this.backend.concat2D(a, b, axis);
  }

  /**
   * Concatenates two 3D ndarrays along a given axis.
   *
   * For example, if:
   * A: shape(2, 1, 3) = | r1, g1, b1 |
   *                     | r2, g2, b2 |
   *
   * B: shape(2, 1, 3) = | r3, g3, b3 |
   *                     | r4, g4, b4 |
   *
   * C = concat3D(A, B, axis)
   *
   * if axis = 0:
   * C: shape(4, 1, 3) = | r1, g1, b1 |
   *                     | r2, g2, b2 |
   *                     | r3, g3, b3 |
   *                     | r4, g4, b4 |
   *
   * if axis = 1:
   * C: shape(2, 2, 3) = | r1, g1, b1, r3, g3, b3 |
   *                     | r2, g2, b2, r4, g4, b4 |
   *
   * if axis = 2:
   * C = shape(2, 1, 6) = | r1, g1, b1, r3, g3, b3 |
   *                      | r2, g2, b2, r4, g4, b4 |
   *
   * @param a The first array to concat.
   * @param b The second array to conat.
   * @param axis The axis to concate along.
   * @return The concatenated array.
   */
  concat3D(a: Array3D, b: Array3D, axis: number): Array3D {
    concat_util.assertParams(a.shape, b.shape, axis);
    return this.backend.concat3D(a, b, axis);
  }

  /**
   * Concatenates two 4D ndarrays along a given axis. See math.concat2D() for
   * documentation.
   *
   * @param a The first array to concat.
   * @param b The second array to conat.
   * @param axis The axis to concate along.
   * @return The concatenated array.
   */
  concat4D(a: Array4D, b: Array4D, axis: number): Array4D {
    concat_util.assertParams(a.shape, b.shape, axis);
    return this.backend.concat4D(a, b, axis);
  }

  ///////////////////
  // Reduction ops //
  ///////////////////

  /**
   * Computes the log(sum(exp(elements across the reduction dimensions)).
   *
   * Reduces the input along the dimensions given in `axis`. Unless `keepDims`
   * is true, the rank of the array is reduced by 1 for each entry in `axis`.
   * If `keepDims` is true, the reduced dimensions are retained with length 1.
   * If `axis` has no entries, all dimensions are reduced, and an array with a
   * single element is returned.
   *
   * @param input The input NDArray.
   * @param axis Optional. The dimension(s) to reduce. If null (the default),
   *     reduces all dimensions.
   * @param keepDims Optional. If true, retains reduced dimensions with length
   *     of 1. Defaults to false.
   */
  logSumExp<T extends NDArray>(
      input: NDArray, axis: number | number[] = null, keepDims = false): T {
    const axes = axis_util.parseAxisParam(axis, input.shape);
    return this.executeOp("logSumExp", () => {
      const xMax = this.max(input, axes, true /* keepDims */);
      const a = this.subtract(input, xMax);
      const b = this.exp(a);
      const c = this.sum(b, axes);
      const d = this.log(c);
      const res = this.add(xMax.reshape(d.shape), d);

      if (keepDims) {
        const newShape = axis_util.expandShapeToKeepDim(res.shape, axes);
        return res.reshape(newShape);
      }
      return res;
    }) as T;
  }

  /**
   * Computes the sum of elements across dimensions of an array.
   *
   * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
   * is true, the rank of the array is reduced by 1 for each entry in `axes`.
   * If `keepDims` is true, the reduced dimensions are retained with length 1.
   * If axes has no entries, all dimensions are reduced, and an array with a
   * single element is returned.
   *
   * @param x The input array to compute the sum over.
   * @param axis Optional. The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims Optional. If true, retains reduced dimensions with size 1.
   */
  sum<D extends DataType, T extends NDArray<SumTypes[D]>>(
      x: NDArray<D>, axis: number | number[] = null, keepDims = false): T {
    const origAxes = axis_util.parseAxisParam(axis, x.shape);
    let axes = origAxes;
    const permutedAxes = axis_util.getPermutedAxes(axes, x.rank);
    return this.executeOp("sum", () => {
      if (permutedAxes != null) {
        x = this.transpose(x, permutedAxes);
        axes = axis_util.getInnerMostAxes(axes.length, x.rank);
      }
      const res = this.backend.sum(x, axes);
      if (keepDims) {
        const newShape = axis_util.expandShapeToKeepDim(res.shape, origAxes);
        return res.reshape(newShape);
      }
      return res;
    }) as T;
  }

  /**
   * Computes the mean of elements across dimensions of an array.
   *
   * Reduces `x` along the dimensions given in `axis`. Unless `keepDims` is
   * true, the rank of the array is reduced by 1 for each entry in `axis`.
   * If `keepDims` is true, the reduced dimensions are retained with length 1.
   * If `axis` has no entries, all dimensions are reduced, and an array with a
   * single element is returned.
   *
   * @param x The input array.
   * @param axis Optional. The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims Optional. If true, retains reduced dimensions with size 1.
   */
  mean(x: NDArray, axis: number | number[] = null, keepDims = false):
      NDArray<"float32"> {
    const axes = axis_util.parseAxisParam(axis, x.shape);
    const shapes = axis_util.computeOutAndReduceShapes(x.shape, axes);
    const reduceShape = shapes[1];
    const reduceSize = util.sizeFromShape(reduceShape);
    return this.executeOp("mean", () => {
      return this.scope(() => {
        const res = this.divide(x, Scalar.new(reduceSize));
        return this.sum(res, axis, keepDims);
      });
    });
  }

  /**
   * Returns the indices of the minimum values along an `axis`. The result has
   * the same shape as `input` with the dimension along `axis` removed.
   *
   * @param x The input array.
   * @param axis Optional. The dimension to reduce. By default it reduces
   * across all axes and returns the flat index.
   *
   */
  argMin<T extends NDArray<"int32">>(x: NDArray, axis: number = null): T {
    let axes = axis_util.parseAxisParam(axis, x.shape);
    const permutedAxes = axis_util.getPermutedAxes(axes, x.rank);
    return this.executeOp("argMin", () => {
      if (permutedAxes != null) {
        x = this.transpose(x, permutedAxes);
        axes = axis_util.getInnerMostAxes(axes.length, x.rank);
      }
      return this.backend.argMin(x, axes);
    }) as T;
  }

  /**
   * Returns the indices of the maximum values along an `axis`. The result has
   * the same shape as `input` with the dimension along `axis` removed.
   *
   * @param x The input array.
   * @param axis Optional. The dimension to reduce. By default it reduces
   *     across all axes and returns the flat index
   */
  argMax<T extends NDArray<"int32">>(x: NDArray, axis: number = null): T {
    let axes = axis_util.parseAxisParam(axis, x.shape);
    const permutedAxes = axis_util.getPermutedAxes(axes, x.rank);
    return this.executeOp("argMax", () => {
      if (permutedAxes != null) {
        x = this.transpose(x, permutedAxes);
        axes = axis_util.getInnerMostAxes(axes.length, x.rank);
      }
      return this.backend.argMax(x, axes);
    }) as T;
  }

  /**
   * Returns a 1 if the argMax of x1 and x2 are the same, otherwise 0.
   * @param x1 The first input NDArray.
   * @param x2 The second input NDArray.
   */
  argMaxEquals(x1: NDArray, x2: NDArray): Scalar<"bool"> {
    util.assertShapesMatch(x1.shape, x2.shape, "Error in argMaxEquals: ");
    return this.executeOp("argMaxEquals", () => this.scope(() => {
      return this.equal(this.argMax(x1), this.argMax(x2));
    }));
  }

  /**
   * Returns the truth value of (a == b) element-wise. Supports broadcasting.
   * For a stricter version without broadcasting use math.equalStrict().
   *
   * @param a The first input `NDArray`.
   * @param b The second input `NDArray`. Must have the same dtype as `a`.
   */
  equal<D1 extends DataType, D2 extends D1, T extends NDArray<"bool">>(
      a: NDArray<D1>, b: NDArray<D2>): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return this.backend.equal(a, b) as T;
  }

  equalStrict<T extends NDArray>(a: T, b: T): NDArray<"bool"> {
    util.assertShapesMatch(a.shape, b.shape, "Error in equalStrict: ");
    return this.equal(a, b);
  }

  /**
   * Returns the truth value of (a != b) element-wise. Supports broadcasting.
   * For a stricter version without broadcasting use math.notEqualStrict().
   *
   * @param a The first input `NDArray`.
   * @param b The second input `NDArray`. Must have the same dtype as `a`.
   */
  notEqual<D1 extends DataType, D2 extends D1, T extends NDArray<"bool">>(
      a: NDArray<D1>, b: NDArray<D2>): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return this.backend.notEqual(a, b) as T;
  }

  notEqualStrict<R extends Rank, D1 extends DataType, D2 extends D1>(
      a: NDArray<D1, R>, b: NDArray<D2, R>): RankMap<"bool">[R] {
    util.assertShapesMatch(a.shape, b.shape, "Error in notEqualStrict: ");
    return this.notEqual(a, b);
  }

  /**
   * Returns the truth value of (a > b) element-wise. Supports broadcasting.
   */
  greater(a: NDArray, b: NDArray): NDArray<"bool"> {
    return this.backend.greater(a, b);
  }

  /**
   * Returns the truth value of (a >= b) element-wise. Supports broadcasting.
   */
  greaterEqual(a: NDArray, b: NDArray): NDArray<"bool"> {
    return this.backend.greaterEqual(a, b);
  }

  /**
   * Returns the truth value of (a < b) element-wise. Supports broadcasting.
   */
  less(a: NDArray, b: NDArray): NDArray<"bool"> {
    return this.backend.less(a, b);
  }

  /**
   * Returns the truth value of (a <= b) element-wise. Supports broadcasting.
   */
  lessEqual(a: NDArray, b: NDArray): NDArray<"bool"> {
    return this.backend.lessEqual(a, b);
  }

  /**
   * Selects elements from `a` or `b`, depending on `cond`.
   */
  select(cond: NDArray<"bool">, a: NDArray, b: NDArray): NDArray {
    return this.backend.select(cond, a, b);
  }

  /**
   * Computes the top K values and flattened indices.
   * @param x The input NDArray.
   * @param k How many top values to compute.
   */
  topK(x: NDArray, k: number): {values: Array1D, indices: Array1D<"int32">} {
    util.assert(
        k <= x.size,
        `Error in topK: k value (${k}) must be less than size of input ` +
            `ndarray, got shape ${x.shape}.`);
    let values: Array1D;
    let indices: Array1D<"int32">;
    this.executeOp("topK", () => {
      values = this.backend.topKValues(x, k);
      indices = this.backend.topKIndices(x, k);
      return values;
    });
    const result = {values, indices};
    return result;
  }

  /**
   * Computes the minimum value from the input.
   *
   * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
   * is true, the rank of the array is reduced by 1 for each entry in `axes`.
   * If `keepDims` is true, the reduced dimensions are retained with length 1.
   * If `axes` has no entries, all dimensions are reduced, and an array with a
   * single element is returned.
   *
   * @param x The input NDArray.
   * @param axis Optional. The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims Optional. If true, retains reduced dimensions with size 1.
   */
  min<D extends DataType, T extends NDArray<D>>(
      x: NDArray<D>, axis: number | number[] = null, keepDims = false): T {
    const origAxes = axis_util.parseAxisParam(axis, x.shape);
    let axes = origAxes;
    const permutedAxes = axis_util.getPermutedAxes(axes, x.rank);
    return this.executeOp("min", () => {
      if (permutedAxes != null) {
        x = this.transpose(x, permutedAxes);
        axes = axis_util.getInnerMostAxes(axes.length, x.rank);
      }
      const res = this.backend.min(x, axes) as NDArray<D>;
      if (keepDims) {
        const newShape = axis_util.expandShapeToKeepDim(res.shape, origAxes);
        return res.reshape(newShape);
      }
      return res;
    }) as T;
  }

  /**
   * Returns the min of a and b (`a < b ? a : b`) element-wise.
   * Supports broadcasting.
   *
   * @param a The first ndarray.
   * @param b The second ndarray. Must have the same type as `a`.
   */
  minimum<D1 extends DataType, D2 extends D1, T extends NDArray<D1>>(
      a: NDArray<D1>, b: NDArray<D2>): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return this.backend.minimum(a, b) as T;
  }

  /**
   * Computes the maximum of elements across dimensions of an array.
   *
   * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
   * is true, the rank of the array is reduced by 1 for each entry in `axes`.
   * If `keepDims` is true, the reduced dimensions are retained with length 1.
   * If `axes` has no entries, all dimensions are reduced, and an array with a
   * single element is returned.
   *
   * @param x The input array.
   * @param axis Optional. The dimension(s) to reduce. By default it reduces
   *     all dimensions.
   * @param keepDims Optional. If true, retains reduced dimensions with size 1.
   */
  max<D extends DataType, T extends NDArray<D>>(
      x: NDArray<D>, axis: number | number[] = null, keepDims = false): T {
    const origAxes = axis_util.parseAxisParam(axis, x.shape);
    let axes = origAxes;
    const permutedAxes = axis_util.getPermutedAxes(axes, x.rank);
    return this.executeOp("max", () => {
      if (permutedAxes != null) {
        x = this.transpose(x, permutedAxes);
        axes = axis_util.getInnerMostAxes(axes.length, x.rank);
      }
      const res = this.backend.max(x, axes) as NDArray<D>;
      if (keepDims) {
        const newShape = axis_util.expandShapeToKeepDim(res.shape, origAxes);
        return res.reshape(newShape);
      }
      return res;
    }) as T;
  }

  /**
   * Returns the max of a and b (`a > b ? a : b`) element-wise.
   * Supports broadcasting.
   *
   * @param a The first ndarray.
   * @param b The second ndarray. Must have the same type as `a`.
   */
  maximum<D1 extends DataType, D2 extends D1, T extends NDArray<D1>>(
      a: NDArray<D1>, b: NDArray<D2>): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return this.backend.maximum(a, b) as T;
  }

  /**
   * Computes the softmax normalized vector given the logits.
   * @param logits The logits array.
   * @param dim The dimension softmax would be performed on. Defaults to -1
   *     which indicates the last dimension.
   */
  softmax<T extends NDArray>(logits: T, dim = -1): T {
    if (dim === -1) {
      dim = logits.rank - 1;
    }
    if (dim !== logits.rank - 1) {
      throw Error(
          "Softmax along a non-last dimension is not yet supported. " +
          `Logits was rank ${logits.rank} and dim was ${dim}`);
    }

    return this.executeOp("softmax", () => {
      // Do it in log space for numerical stability.
      // exp(X - logSumExp(X))
      const keepDims = true;
      const lse = this.logSumExp(logits, [dim], keepDims);
      const logResult = this.subtract(logits, lse);
      const value = this.exp(logResult) as T;
      return value;
    });
  }

  /**
   * Computes softmax cross entropy between logits and labels.
   *
   * Measures the probability error in discrete classification tasks in which
   * the classes are mutually exclusive (each entry is in exactly one class).
   * For example, each CIFAR-10 image is labeled with one and only one label: an
   * image can be a dog or a truck, but not both.
   *
   * NOTE: While the classes are mutually exclusive, their probabilities need
   * not be. All that is required is that each row of labels is a valid
   * probability distribution. If they are not, the computation of the gradient
   * will be incorrect.
   *
   * WARNING: This op expects unscaled logits, since it performs a softmax on
   * logits internally for efficiency. Do not call this op with the output of
   * softmax, as it will produce incorrect results.
   *
   * logits and labels must have the same shape, e.g. [batch_size, num_classes]
   * and the same dtype.
   * @param labels The labels array.
   * @param logits The logits array.
   * @param dim The dimension softmax would be performed on. Defaults to -1
   *     which indicates the last dimension.
   */
  softmaxCrossEntropyWithLogits<I extends NDArray<"float32">,
                                          O extends NDArray<"float32">>(
      labels: I, logits: I, dim = -1): O {
    util.assertShapesMatch(
        labels.shape, logits.shape, "Error in softmaxCrossEntropyWithLogits: ");
    if (dim === -1) {
      dim = logits.rank - 1;
    }
    if (dim !== logits.rank - 1) {
      throw Error(
          `Softmax cross entropy along a non-last dimension is not yet ` +
          `supported. Labels / logits was rank ${logits.rank} ` +
          `and dim was ${dim}`);
    }

    return this.executeOp("softmaxCrossEntropyWithLogits", () => {
      const softmaxLogits = this.softmax(logits, dim);
      const yPlusEps = this.add(Scalar.new(1e-5), softmaxLogits);
      const logOutput = this.log(yPlusEps);
      const tarLogOutput = this.multiply(labels, logOutput);
      const costVector = this.neg(tarLogOutput);
      const value = this.sum(costVector, [dim]) as O;
      return value;
    });
  }

  //////////////////////
  // Element-wise ops //
  //////////////////////

  /** @deprecated Use math.transpose() instead. */
  switchDim<T extends NDArray>(a: T, newDim: number[]): T {
    return this.transpose(a, newDim);
  }

  /**
   * Construct an array by repeating it the number of times given by reps.
   *
   * This operation creates a new array by replicating `input` `reps`
   * times. The output tensor's i'th dimension has `input.shape[i] *
   * reps[i]` elements, and the values of `input` are replicated
   * `reps[i]` times along the i'th dimension. For example, tiling
   * `[a, b, c, d]` by `[2]` produces `[a, b, c, d, a, b, c, d]`.
   *
   * @param x The array to transpose.
   * @param reps Determines the number of replications per dimension.
   */
  tile<D extends DataType, T extends NDArray<D>>(x: T, reps: number[]): T {
    util.assert(
        x.rank === reps.length,
        `Error in transpose: rank of input ${x.rank} ` +
            `must match length of reps ${reps}.`);
    return this.backend.tile(x, reps) as T;
  }

  /**
   * Transposes the array. Permutes the dimensions according to `perm`.
   *
   * The returned array's dimension `i` will correspond to the input dimension
   * `perm[i]`. If `perm` is not given, it is set to `[n-1...0]`, where `n` is
   * the rank of the input array. Hence by default, this operation performs a
   * regular matrix transpose on 2-D input arrays.
   *
   * @param x The array to transpose.
   * @param perm Optional. The permutation of the dimensions of a.
   */
  transpose<T extends NDArray>(x: T, perm?: number[]): T {
    if (perm == null) {
      perm = x.shape.map((s, i) => i).reverse();
    }
    util.assert(
        x.rank === perm.length,
        `Error in transpose: rank of input ${x.rank} ` +
            `must match length of perm ${perm}.`);
    return this.backend.transpose(x, perm) as T;
  }

  /** @deprecated Use math.add(c, A) instead. */
  scalarPlusArray<T extends NDArray>(c: Scalar, a: T): T {
    util.assert(
        c.size === 1,
        `Error in scalarPlusArray: first argument must be rank 0, but got ` +
            `rank ${c.rank}.`);
    return this.add(c, a) as T;
  }

  /** @deprecated Use math.sub(c, A) instead. */
  scalarMinusArray<T extends NDArray>(c: Scalar, a: T): T {
    util.assert(
        c.size === 1,
        `Error in scalarMinusArray: first argument must be rank 0, but got ` +
            `rank ${c.rank}.`);
    return this.subtract(c, a) as T;
  }

  /** @deprecated Use math.sub(A, c) instead. */
  arrayMinusScalar<T extends NDArray>(a: T, c: Scalar): T {
    util.assert(
        c.size === 1,
        `Error in arrayMinusScalar: second argument must be rank 0, but ` +
            `got rank ${c.rank}.`);
    return this.subtract(a, c) as T;
  }

  /**
   * Computes -1 * A element-wise.
   * @param x The input array.
   */
  neg<T extends NDArray>(x: T): T {
    return this.backend.neg(x) as T;
  }

  /**
   * Adds two NDArrays element-wise, A + B. Supports broadcasting.
   * For a stricter version without broadcasting use math.addStrict().
   *
   * @param a The first `NDArray` to add.
   * @param b The second `NDArray` to add. Must have the same type as `a`.
   */
  add<D1 extends DataType, D2 extends D1, T extends NDArray<D1>>(
      a: NDArray<D1>, b: NDArray<D2>): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return this.backend.add(a, b) as T;
  }

  /**
   * Adds two NDArrays element-wise, A + B. Inputs must
   * be the same shape. For broadcasting support, use math.add() instead.
   *
   * @param a The first NDArray to multiply element-wise.
   * @param b The second NDArray to multiply element-wise.
   */
  addStrict<T extends NDArray>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, "Error in addStrict: ");
    return this.add(a, b) as T;
  }

  /**
   * Subtracts two NDArrays element-wise, A - B. Supports broadcasting.
   * For a stricter version without broadcasting use math.subStrict().
   *
   * @param a The first `NDArray`.
   * @param b The second `NDArray`. Must have the same dtype as `a`.
   */
  subtract<D1 extends DataType, D2 extends D1, T extends NDArray<D1>>(
      a: NDArray<D1>, b: NDArray<D2>): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return this.backend.subtract(a, b) as T;
  }

  /**
   * Computes the power of one value to another.
   * Given a tensor x and a tensor y, this operation computes x^y for
   * corresponding elements in x and y. For example:
   * x = tf.constant([[2, 2], [3, 3]])
   * y = tf.constant([[8, 16], [2, 3]])
   * pow(x, y)  # [[256, 65536], [9, 27]]
   *
   * @param a The base NDArray to pow element-wise.
   * @param b The exponent NDArray to pow element-wise.
   */
  pow<D extends DataType, T extends NDArray<D>>(
      a: NDArray<D>, b: NDArray): T {
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return this.backend.pow(a, b) as T;
  }

  /**
   * Computes the power of one value to another. Inputs must
   * be the same shape. For broadcasting support, use math.pow() instead.
   *
   * @param a The base NDArray to pow element-wise.
   * @param b The exponent NDArray to pow element-wise.
   */
  powStrict<D extends DataType>(a: NDArray<D>, b: NDArray):
      NDArray<D> {
    util.assertShapesMatch(a.shape, b.shape, "Error in powStrict: ");
    return this.pow(a, b);
  }

  /** @deprecated Use math.subtract instead. */
  sub<D1 extends DataType, D2 extends D1, T extends NDArray<D1>>(
      a: NDArray<D1>, b: NDArray<D2>): T {
    return this.subtract(a, b);
  }

  /**
   * Subtracts two NDArrays element-wise, A - B. Inputs must
   * be the same shape. For broadcasting support, use math.sub() instead.
   *
   * @param a The first NDArray to multiply element-wise.
   * @param b The second NDArray to multiply element-wise.
   */
  subStrict<T extends NDArray>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, "Error in subStrict: ");
    return this.subtract(a, b);
  }

  /**
   * Multiplies two NDArrays element-wise, A * B. Supports broadcasting.
   * For a stricter version without broadcasting use math.multiplyStrict().
   *
   * @param a The first `NDArray`.
   * @param b The second `NDArray`. Must have the same dtype as `a`.
   */
  multiply<D1 extends DataType, D2 extends D1, T extends NDArray<D1>>(
      a: NDArray<D1>, b: NDArray<D2>): T {
    util.assertTypesMatch(a, b);
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return this.backend.multiply(a, b) as T;
  }

  /**
   * @deprecated Use math.multiplyStrict() instead.
   */
  elementWiseMul<T extends NDArray>(a: T, b: T): T {
    return this.multiplyStrict(a, b);
  }

  /**
   * Multiplies two NDArrays element-wise, A * B. Inputs must
   * be the same shape. For broadcasting support, use math.multiply() instead.
   *
   * @param a The first NDArray to multiply element-wise.
   * @param b The second NDArray to multiply element-wise.
   */
  multiplyStrict<T extends NDArray>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, "Error in multiplyStrict: ");
    return this.multiply(a, b) as T;
  }

  /**
   * Divides two NDArrays element-wise, A / B. Supports broadcasting.
   * For a stricter version without broadcasting use math.divideStrict().
   *
   * @param a The first NDArray to divide element-wise.
   * @param b The second NDArray to divide element-wise.
   */
  divide<T extends NDArray<"float32">>(a: NDArray, b: NDArray): T {
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return this.backend.divide(a, b) as T;
  }

  /**
   * Divides two NDArrays element-wise, A / B. Inputs must
   * be the same shape. For broadcasting support, use math.divide() instead.
   *
   * @param a The first NDArray to multiply element-wise.
   * @param b The second NDArray to multiply element-wise.
   */
  divideStrict<T extends NDArray>(a: T, b: T): T {
    util.assertShapesMatch(a.shape, b.shape, "Error in divideStrict: ");
    return this.divide(a, b) as T;
  }

  /** @deprecated Use math.divide(c, A) instead. */
  scalarDividedByArray<T extends NDArray>(c: Scalar, a: T): T {
    util.assert(
        c.size === 1,
        `Error in scalarDividedByArray: first argument must be rank 0, but ` +
            `got NDArray of rank ${c.rank}.`);
    return this.divide(c, a) as T;
  }

  /** @deprecated Use math.divide(A, c) instead. */
  arrayDividedByScalar<T extends NDArray>(a: T, c: Scalar): T {
    util.assert(
        c.size === 1,
        `Error in arrayDividedByScalar: second argument must be rank 0, ` +
            `but got NDArray of rank ${c.rank}.`);
    return this.divide(a, c) as T;
  }

  /**
   * Computes ceiling of input NDArray element-wise. y = ceil(x)
   * TODO(nsthorat): Make this return an int32 when we add rank as a
   * generic.
   * @param x The input NDArray.
   */
  ceil<T extends NDArray>(x: T): T {
    return this.backend.ceil(x) as T;
  }

  /**
   * Computes floor of input NDArray element-wise. y = floor(x).
   *
   * @param x The input NDArray.
   */
  floor<T extends NDArray>(x: T): T {
    return this.backend.floor(x) as T;
  }

  /**
   * Computes exponential of the input NDArray element-wise. y = e ^ x
   * @param x The input NDArray.
   */
  exp<T extends NDArray>(x: T): T {
    return this.backend.exp(x) as T;
  }

  /**
   * Computes natural logarithm of the input NDArray element-wise. y = ln(x)
   * @param x The input NDArray.
   */
  log<T extends NDArray>(x: T): T {
    return this.backend.log(x) as T;
  }

  /**
   * Computes square root of the input NDArray element-wise. y = sqrt(x)
   * @param x The input NDArray.
   */
  sqrt<T extends NDArray>(x: T): T {
    return this.backend.sqrt(x) as T;
  }

  /**
   * Computes square of `x` element-wise.
   *
   * @param x The input array.
   */
  square<T extends NDArray>(x: T): T {
    return this.backend.square(x) as T;
  }

  /**
   * Computes absolute value element-wise.
   * @param x The input NDArray.
   */
  abs<T extends NDArray>(x: T): T {
    return this.backend.abs(x) as T;
  }

  /**
   * Clips values element-wise.
   * @param x The input NDArray.
   * @param min Lower-bound of range to be clipped to.
   * @param max Upper-bound of range to be clipped to.
   */
  clip<T extends NDArray>(x: T, min: number, max: number): T {
    util.assert(
        (min <= max),
        `Error in clip: min (${min}) must be` +
            `less than or equal to max (${max}).`);
    return this.backend.clip(x, min, max) as T;
  }

  /**
   * Computes rectified linear element-wise, max(x, 0).
   * @param x The input NDArray.
   */
  relu<T extends NDArray>(x: T): T {
    return this.backend.relu(x) as T;
  }

  /**
   * Computes exponential linear element-wise
   * @param {T} x the input NDArray
   */
  elu<T extends NDArray>(x: T): T {
    return this.backend.elu(x) as T;
  }

  /**
   * Computes the derivative of elu which is used ly
   * @hidden
   */
  eluDer<T extends NDArray>(x: T): T {
    return this.backend.eluDer(x) as T;
  }

  /**
   * Computes scaled exponential linear element-wise.
   * @hidden
   */
  selu<T extends NDArray>(x: T): T {
    return this.backend.selu(x) as T;
  }

  /**
   * Computes leaky rectified linear element-wise
   * @param {T} x the input NDArray
   * @param alpha scaling factor for negative values, defaults to 0.2
   * @return {NDArray}
   */
  leakyRelu<T extends NDArray>(x: T, alpha = 0.2): T {
    return this.backend.leakyRelu(x, alpha) as T;
  }

  /**
   * Computes leaky rectified linear element-wise with parametric alphas
   * @param {T} x the input NDArray
   * @param {T} alpha scaling factor NDArray for negative values
   * @return {NDArray}
   */
  prelu<T extends NDArray>(x: T, alpha: T): T {
    return this.backend.prelu(x, alpha) as T;
  }

  /**
   * Computes the derivative of PReLU
   * @param {T} x the input NDArray
   * @param {T} alpha scaling factor NDArray for negative values
   * @return {NDArray}
   */
  preluDer<T extends NDArray>(x: T, alpha: T): T {
    return this.backend.preluDer(x, alpha) as
        T;
  }

  /**
   * Computes sigmoid element-wise, y = 1 / (1 + exp(-x)).
   * @param x The input NDArray.
   */
  sigmoid<T extends NDArray>(x: T): T {
    return this.backend.sigmoid(x) as T;
  }

  /**
   * Computes sin of the input NDArray element-wise, y = sin(x).
   * @param x The input NDArray.
   */
  sin<T extends NDArray>(x: T): T {
    return this.backend.sin(x) as T;
  }

  /**
   * Computes cos of the input NDArray element-wise, y = cos(x).
   * @param x The input NDArray.
   */
  cos<T extends NDArray>(x: T): T {
    return this.backend.cos(x) as T;
  }

  /**
   * Computes tan of the input NDArray element-wise, y = tan(x).
   * @param x The input NDArray.
   */
  tan<T extends NDArray>(x: T): T {
    return this.backend.tan(x) as T;
  }

  /**
   * Computes asin of the input NDArray element-wise, y = asin(x).
   * @param x The input NDArray.
   */
  asin<T extends NDArray>(x: T): T {
    return this.backend.asin(x) as T;
  }

  /**
   * Computes acos of the input NDArray element-wise, y = acos(x).
   * @param x The input NDArray.
   */
  acos<T extends NDArray>(x: T): T {
    return this.backend.acos(x) as T;
  }

  /**
   * Computes atan of the input NDArray element-wise, y = atan(x).
   * @param x The input NDArray.
   */
  atan<T extends NDArray>(x: T): T {
    return this.backend.atan(x) as T;
  }

  /**
   * Computes hyperbolic sin of the input NDArray element-wise, y = sinh(x).
   * @param x The input NDArray.
   */
  sinh<T extends NDArray>(x: T): T {
    return this.backend.sinh(x) as T;
  }

  /**
   * Computes hyperbolic cos of the input NDArray element-wise, y = cosh(x).
   * @param x The input NDArray.
   */
  cosh<T extends NDArray>(x: T): T {
    return this.backend.cosh(x) as T;
  }

  /**
   * Computes hyperbolic tangent of the input NDArray element-wise.
   * @param x The input NDArray.
   */
  tanh<T extends NDArray>(x: T): T {
    return this.backend.tanh(x) as T;
  }

  /**
   * Computes step of the input NDArray element-wise,
   * y=1 if x>0|alpha*x if x<=0.
   *
   * @param x The input NDArray.
   * @param alpha The gradient when input is negative.
   */
  step<T extends NDArray>(x: T, alpha = 0.0): T {
    return this.backend.step(x, alpha) as T;
  }

  /**
   * Computes a scaled array add operation, c1 * A + c2 * B.
   * @param c1 The first scalar in the scaled array add computation.
   * @param a The first NDArray in the scaled array add computation.
   * @param c2 The second scalar in the scaled array add computation.
   * @param cb The second NDArray in the scaled array add computation.
   */
  scaledArrayAdd<T extends NDArray>(c1: Scalar, a: T, c2: Scalar, b: T): T {
    util.assert(
        c1.size === 1,
        `Error in scaledArrayAdd: first argument must rank 0, but got ` +
            ` rank ${c1.rank}.`);
    util.assert(
        c2.size === 1,
        `Error in scaledArrayAdd: third argument must be rank 0, but got ` +
            `NDArray of rank ${c2.rank}.`);
    util.assertShapesMatch(a.shape, b.shape, "Error in scaledArrayAdd: ");

    return this.executeOp("scaledArrayAdd", () => {
      return this.scope(() => {
        // TODO(nsthorat): Add an SGEMM kernel and then update this.
        return this.add(this.multiply(c1, a), this.multiply(c2, b)) as T;
      });
    });
  }

  /** @deprecated Use math.multiply(c, A) instead. */
  scalarTimesArray<T extends NDArray>(c: Scalar, a: T): T {
    util.assert(
        c.size === 1,
        `Error in arrayDividedByScalar: first argument must be rank 0, but ` +
            `got rank ${c.rank}.`);
    return this.multiply(c, a) as T;
  }

  /**
   * @deprecated Use math.multiply() instead.
   */
  elementWiseMulBroadcast(a: Array2D, b: Array2D): Array2D {
    util.assert(
        a.rank === 2,
        `Error in elementWiseMulBroadcast: first argument must be ` +
            `rank 2, but got rank ${a.rank}.`);
    util.assert(
        b.rank === 2,
        `Error in elementWiseMulBroadcast: second argument must be ` +
            `rank 2, but got rank ${b.rank}.`);
    return this.multiply(a, b) as Array2D;
  }

  /////////////////////
  // Convolution ops //
  /////////////////////

  /**
   * Computes a 1D convolution over the input x.
   * @param input The input ndarray, of rank 3 or rank 2, of shape
   *     `[batch, width, inChannels]`. If rank 2, batch of 1 is assumed.
   * @param filter The filter, rank 3, of shape
   *     [filterWidth, inDepth, outDepth].
   * @param bias Optional bias, rank 1 of shape [outDepth].
   * @param stride The number of entries by which the filter is moved right at
   *     each step.
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm.
   *    - 'same' pad and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - 'valid' pad: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     https://www.tensorflow.org/api_guides/python/nn#Convolution
   */
  conv1d<T extends NDArray>(
      input: T, filter: Array3D, bias: Array1D | null, stride: number,
      pad: "valid" | "same" | number): T {
    let input3D = input as NDArray as Array3D;
    let reshapedTo3D = false;
    if (input.rank === 2) {
      reshapedTo3D = true;
      input3D = input.as3D(1, input.shape[0], input.shape[1]);
    }

    util.assert(
        input3D.rank === 3,
        `Error in conv1d: input must be rank 3, but got rank ${input3D.rank}.`);
    util.assert(
        filter.rank === 3,
        `Error in conv1d: filter must be rank 3, but got rank ` +
            `${filter.rank}.`);
    if (bias != null) {
      util.assert(
          bias.rank === 1,
          `Error in conv1d: bias must be rank 1, but got rank ` +
              `${bias.rank}.`);
    }

    util.assert(
        input3D.shape[2] === filter.shape[1],
        `Error in conv1d: depth of input (${input3D.shape[2]}) must match  ` +
            `input depth for filter ${filter.shape[1]}.`);

    const filter4D =
        filter.as4D(1, filter.shape[0], filter.shape[1], filter.shape[2]);
    const input4D =
        input3D.as4D(input3D.shape[0], 1, input3D.shape[1], input3D.shape[2]);
    const strides: [number, number] = [1, stride];

    return this.executeOp("Conv1D", () => {
      const res = this.conv2d(input4D, filter4D, bias, strides, pad);
      if (reshapedTo3D) {
        return res.as2D(res.shape[2], res.shape[3]) as NDArray as T;
      }
      return res.as3D(res.shape[0], res.shape[2], res.shape[3]) as NDArray as T;
    });
  }

  /**
   * Computes a 2D convolution over the input x.
   *
   * @param input The input ndarray, of rank 4 or rank 3, of shape
   *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
   * assumed.
   * @param filter The filter, rank 4, of shape
   *     [filterHeight, filterWidth, inDepth, outDepth].
   * @param bias Optional bias, rank 1 of shape [outDepth].
   * @param strides The strides of the convolution: [strideHeight,
   * strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm.
   *    - 'same' pad and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - 'valid' pad: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     https://www.tensorflow.org/api_guides/python/nn#Convolution
   */
  conv2d<T extends NDArray>(
      input: T, filter: Array4D, bias: Array1D | null,
      strides: [number, number] | number, pad: "valid" | "same" | number): T {
    let input4D = input as NDArray as Array4D;
    let reshapedTo4D = false;
    if (input.rank === 3) {
      reshapedTo4D = true;
      input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
    }
    util.assert(
        input4D.rank === 4,
        `Error in conv2d: input must be rank 4, but got rank ${input4D.rank}.`);
    util.assert(
        filter.rank === 4,
        `Error in conv2d: filter must be rank 4, but got rank ` +
            `${filter.rank}.`);
    if (bias != null) {
      util.assert(
          bias.rank === 1,
          `Error in conv2d: bias must be rank 1, but got rank ` +
              `${bias.rank}.`);
    }

    util.assert(
        input4D.shape[3] === filter.shape[2],
        `Error in conv2d: depth of input (${input4D.shape[3]}) must match  ` +
            `input depth for filter ${filter.shape[2]}.`);

    const convInfo =
        conv_util.computeConv2DInfo(input4D.shape, filter.shape, strides, pad);
    return this.executeOp("Conv2D", () => {
      const res = this.backend.conv2d(input4D, filter, bias, convInfo);
      if (reshapedTo4D) {
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as NDArray as
            T;
      }
      return res as NDArray as T;
    });
  }

  /**
   * Computes the derivative of the input of a 2D convolution.
   *
   * @param inShape The shape of the input: [batch, height, width, inDepth].
   * If length of 3, batch of 1 is assumed.
   * @param dy The derivative of the output, of rank 4 or rank 3 of shape
   *   [batch, outHeight, outWidth, outDepth]. If rank 3, batch of 1 is
   * assumed.
   * @param filter The filter, rank 4, of shape
   *     [filterHeight, filterWidth, inDepth, outDepth].
   * @param strides The strides of the convolution: [strideHeight,
   * strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm
   *     used in the forward prop of the op.
   */
  conv2dDerInput<T extends NDArray>(
      inShape: [number, number, number, number] | [number, number, number],
      dy: T, filter: Array4D, strides: [number, number] | number,
      pad: "valid" | "same" | number): T {
    util.assert(
        inShape.length === dy.rank,
        `Length of inShape ` +
            `(${inShape.length}) and rank of dy (${dy.rank}) must match`);

    let inShape4D = inShape as [number, number, number, number];
    let dy4D = dy as NDArray as Array4D;
    let reshapedTo4D = false;
    if (dy.rank === 3) {
      reshapedTo4D = true;
      dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
      inShape4D = [1, inShape[0], inShape[1], inShape[2]];
    }

    const inDepth = inShape4D[3];
    const outDepth = dy4D.shape[3];
    util.assert(
        inShape4D.length === 4,
        `Error in conv2dDerInput: inShape must be length 4, but got length ` +
            `${inShape4D.length}.`);
    util.assert(
        dy4D.rank === 4,
        `Error in conv2dDerInput: dy must be rank 4, but got ` +
            `rank ${dy4D.rank}`);
    util.assert(
        filter.rank === 4,
        `Error in conv2dDerInput: filter must be rank 4, but got ` +
            `rank ${filter.rank}`);
    util.assert(
        inDepth === filter.shape[2],
        `Error in conv2dDerInput: depth of input (${inDepth}) must ` +
            `match input depth for filter ${filter.shape[2]}.`);
    util.assert(
        outDepth === filter.shape[3],
        `Error in conv2dDerInput: depth of output (${outDepth}) must` +
            `match output depth for filter ${filter.shape[3]}.`);

    const convInfo =
        conv_util.computeConv2DInfo(inShape4D, filter.shape, strides, pad);
    return this.executeOp("conv2dDerInput", () => {
      const res = this.backend.conv2dDerInput(dy4D, filter, convInfo);
      if (reshapedTo4D) {
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as NDArray as
            T;
      }
      return res as NDArray as T;
    });
  }

  /**
   * Computes the derivative of the bias of a 2D convolution.
   *
   * @param dy The gradient for the output of this op, of rank 4 or rank 3 of
   *   shape [batch, height, width, outDepth]. If rank 3, batch of 1 is
   * assumed.
   */
  conv2dDerBias(dy: Array3D | Array4D): Array1D {
    let dy4D = dy as Array4D;
    if (dy.rank === 3) {
      dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
    }
    return this.backend.conv2dDerBias(dy4D);
  }

  /**
   * Computes the derivative of the filter of a 2D convolution.
   *
   * @param input The input ndarray, of rank 4 or rank 3 of shape
   *     [batch, height, width, inChannels]. If rank 3, batch of 1 is assumed.
   * @param dy The dy image, of rank 4 or rank 3, of shape
   *     [batch, height, width, outDepth]. If rank 3, batch of 1 is assumed.
   * @param filterShape The shape of the filter, length 4,
   *     [filterHeight, filterWidth, inDepth, outDepth].
   * @param strides The strides of the convolution: [strideHeight,
   * strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm
   *     used in the forward prop of the op.
   */
  conv2dDerFilter<T extends NDArray>(
      input: T, dy: T, filterShape: [number, number, number, number],
      strides: [number, number] | number, pad: "valid" | "same" | number):
      Array4D {
    let input4D = input as NDArray as Array4D;
    if (input.rank === 3) {
      input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
    }
    let dy4D = dy as NDArray as Array4D;
    if (dy4D.rank === 3) {
      dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
    }
    util.assert(
        input4D.rank === 4,
        `Error in conv2dDerFilter: input must be rank 4, but got shape ` +
            `${input4D.shape}.`);
    util.assert(
        dy4D.rank === 4,
        `Error in conv2dDerFilter: dy must be rank 4, but got shape ` +
            `${dy4D.shape}.`);
    util.assert(
        filterShape.length === 4,
        `Error in conv2dDerFilter: filterShape must be length 4, but got ` +
            `${filterShape}.`);
    util.assert(
        input4D.shape[3] === filterShape[2],
        `Error in conv2dDerFilter: depth of input ${input4D.shape[3]}) must ` +
            `match input depth in filter (${filterShape[2]}.`);
    util.assert(
        dy4D.shape[3] === filterShape[3],
        `Error in conv2dDerFilter: depth of dy (${dy4D.shape[3]}) must ` +
            `match output depth for filter (${filterShape[3]}).`);

    const convInfo =
        conv_util.computeConv2DInfo(input4D.shape, filterShape, strides, pad);
    return this.backend.conv2dDerFilter(input4D, dy4D, convInfo);
  }

  /**
   * Computes the transposed 2D convolution of an image, also known as a
   * deconvolution.
   *
   * @param x The input image, of rank 4 or rank 3, of shape
   *   [batch, height, width, inDepth]. If rank 3, batch of 1 is assumed.
   * @param filter The filter, rank 4, of shape
   *     `[filterHeight, filterWidth, outDepth, inDepth]`.
   *     `inDepth` must match `inDepth` in `x`.
   * @param outputShape Output shape, of rank 4 or rank 3:
   *     [batch, height, width, outDepth]. If rank 3, batch of 1 is assumed.
   * @param strides The strides of the original convolution:
   *     `[strideHeight, strideWidth]`.
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm
   *     used in the non-transpose version of the op.
   */
  conv2dTranspose<T extends NDArray>(
      x: T, filter: Array4D,
      outputShape: [number, number, number, number] | [number, number, number],
      strides: [number, number] | number, pad: "valid" | "same" | number): T {
    return this.conv2dDerInput(outputShape, x, filter, strides, pad);
  }

  /**
   * Depthwise 2D convolution.
   *
   * Given a 4D `input` array and a `filter` array of shape
   * `[filterHeight, filterWidth, inChannels, channelMultiplier]` containing
   * `inChannels` convolutional filters of depth 1, this op applies a
   * different filter to each input channel (expanding from 1 channel to
   * `channelMultiplier` channels for each), then concatenates the results
   * together. The output has `inChannels * channelMultiplier` channels.
   *
   * See https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d for
   * more details.
   *
   * @param input The input ndarray, of rank 4 or rank 3, of shape
   *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
   * assumed.
   * @param filter The filter ndarray, rank 4, of shape
   *     `[filterHeight, filterWidth, inChannels, channelMultiplier]`.
   * @param strides The strides of the convolution: [strideHeight,
   * strideWidth]. If strides is a single number, then `strideHeight ==
   * strideWidth`.
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm.
   *   - 'same' pad and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *   - 'valid' pad: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     https://www.tensorflow.org/api_guides/python/nn#Convolution
   * @param rates The dilation rates: `[rateHeight, rateWidth]` in which we
   *     sample input values across the height and width dimensions in atrous
   *     convolution. Defaults to `[1, 1]`. If `rate` is a single number, then
   *     `rateHeight == rateWidth`. If it is greater than 1, then all values
   * of `strides` must be 1.
   */
  depthwiseConv2D<T extends NDArray>(
      input: T, filter: Array4D, strides: [number, number] | number,
      pad: "valid" | "same" | number,
      rates: [number, number] | number = [1, 1]): T {
    let input4D = input as NDArray as Array4D;
    let reshapedTo4D = false;
    if (input.rank === 3) {
      reshapedTo4D = true;
      input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
    }
    util.assert(
        input4D.rank === 4,
        `Error in depthwiseConv2D: input must be rank 4, but got ` +
            `rank ${input4D.rank}.`);
    util.assert(
        filter.rank === 4,
        `Error in depthwiseConv2D: filter must be rank 4, but got rank ` +
            `${filter.rank}.`);
    util.assert(
        input4D.shape[3] === filter.shape[2],
        `Error in depthwiseConv2D: number of input channels ` +
            `(${input4D.shape[3]}) must match the inChannels dimension in ` +
            `filter ${filter.shape[2]}.`);
    rates = rates || [1, 1];
    const [rateHeight, rateWidth] = parseTupleParam(rates);
    util.assert(
        rateHeight === 1 && rateWidth === 1,
        "Error in depthwiseConv2D: rates greater than 1 are not yet " +
            `supported. Got rates '${rates}'`);

    const convInfo = conv_util.computeConv2DInfo(
        input4D.shape, filter.shape, strides, pad, true /* depthwise */);
    return this.executeOp("depthwiseConv2D", () => {
      const res = this.backend.depthwiseConv2D(input4D, filter, convInfo);
      if (reshapedTo4D) {
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as NDArray as
            T;
      }
      return res as NDArray as T;
    });
  }

  /**
   * Computes the 2D max pooling of an image.
   *
   * @param input The input ndarray, of rank 4 or rank 3 of shape
   *     [batch, height, width, inChannels]. If rank 3, batch of 1 is assumed.
   * @param filterSize The filter size, a tuple [filterHeight, filterWidth].
   * @param strides The strides of the pooling: [strideHeight, strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm.
   *    - 'same' pad and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - 'valid' pad: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     https://www.tensorflow.org/api_guides/python/nn#Convolution
   */
  maxPool<T extends NDArray>(
      input: T, filterSize: [number, number] | number,
      strides: [number, number] | number, pad: "valid" | "same" | number): T {
    let input4D = input as NDArray as Array4D;
    let reshapedTo4D = false;
    if (input.rank === 3) {
      reshapedTo4D = true;
      input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
    }
    util.assert(
        input4D.rank === 4,
        `Error in maxPool: input must be rank 4 but got rank ${input4D.rank}.`);

    const convInfo =
        conv_util.computePool2DInfo(input4D.shape, filterSize, strides, pad);
    return this.executeOp("maxPool", () => {
      const res = this.backend.maxPool(input4D, convInfo);
      if (reshapedTo4D) {
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as NDArray as
            T;
      }
      return res as NDArray as T;
    });
  }

  /**
   * Computes the backprop of a max pool.
   *
   * @param dy The dy error, of rank 4 or rank 3 of shape
   *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
   * assumed.
   * @param input The input image, of rank 4 or rank 3 of shape
   *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
   * assumed.
   * @param filterSize The filter size, a tuple [filterHeight, filterWidth].
   * @param strides The strides of the pooling: [strideHeight, strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm
   *     used in the forward prop of the op.
   */
  maxPoolBackprop<T extends NDArray>(
      dy: T, input: T, filterSize: [number, number] | number,
      strides: [number, number] | number, pad: "valid" | "same" | number): T {
    util.assert(
        input.rank === dy.rank,
        `Rank of input (${input.rank}) does not match rank of dy (${dy.rank})`);

    let input4D = input as NDArray as Array4D;
    let dy4D = dy as NDArray as Array4D;
    let reshapedTo4D = false;
    if (input.rank === 3) {
      reshapedTo4D = true;
      input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
      dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
    }

    util.assert(
        dy4D.rank === 4,
        `Error in maxPoolBackprop: dy must be rank 4 but got rank ` +
            `${dy4D.rank}.`);
    util.assert(
        input4D.rank === 4,
        `Error in maxPoolBackprop: input must be rank 4 but got rank ` +
            `${input4D.rank}.`);

    const convInfo =
        conv_util.computePool2DInfo(input4D.shape, filterSize, strides, pad);
    return this.executeOp("maxPoolBackprop", () => {
      const res = this.backend.maxPoolBackprop(dy4D, input4D, convInfo);
      if (reshapedTo4D) {
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as NDArray as
            T;
      }
      return res as NDArray as T;
    });
  }

  /**
   * Computes the 2D min pooling of an image.
   *
   * @param input The input ndarray, of rank 4 or rank 3 of shape
   *     [batch, height, width, inChannels]. If rank 3, batch of 1 is assumed.
   * @param filterSize The filter size, a tuple [filterHeight, filterWidth].
   * @param strides The strides of the pooling: [strideHeight, strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm.
   *    - 'same' pad and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - 'valid' pad: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     https://www.tensorflow.org/api_guides/python/nn#Convolution
   */
  minPool<T extends NDArray>(
      input: T, filterSize: [number, number] | number,
      strides: [number, number] | number, pad: "valid" | "same" | number): T {
    let input4D = input as NDArray as Array4D;
    let reshapedTo4D = false;
    if (input.rank === 3) {
      reshapedTo4D = true;
      input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
    }
    util.assert(
        input4D.rank === 4,
        `Error in minPool: x must be rank 4 but got rank ${input4D.rank}.`);

    const convInfo =
        conv_util.computePool2DInfo(input4D.shape, filterSize, strides, pad);
    return this.executeOp("minPool", () => {
      const res = this.backend.minPool(input4D, convInfo);
      if (reshapedTo4D) {
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as NDArray as
            T;
      }
      return res as NDArray as T;
    });
  }

  /**
   * Computes the 2D average pooling of an image.
   *
   * @param input The input ndarray, of rank 4 or rank 3 of shape
   *     [batch, height, width, inChannels]. If rank 3, batch of 1 is assumed.
   * @param filterSize The filter size, a tuple [filterHeight, filterWidth].
   * @param strides The strides of the pooling: [strideHeight, strideWidth].
   * @param pad A string from: 'same', 'valid'. The type of padding algorithm.
   *    - 'same' pad and stride 1: output will be of same size as input,
   *       regardless of filter size.
   *    - 'valid' pad: output will be smaller than input if filter is larger
   *       than 1x1.
   *   - For more info, see this guide:
   *     https://www.tensorflow.org/api_guides/python/nn#Convolution
   */
  avgPool<T extends NDArray>(
      input: T, filterSize: [number, number] | number,
      strides: [number, number] | number, pad: "valid" | "same" | number): T {
    let input4D = input as NDArray as Array4D;
    let reshapedTo4D = false;
    if (input.rank === 3) {
      reshapedTo4D = true;
      input4D = input.as4D(1, input.shape[0], input.shape[1], input.shape[2]);
    }
    util.assert(
        input4D.rank === 4,
        `Error in avgPool: x must be rank 4 but got rank ${input4D.rank}.`);

    const convInfo =
        conv_util.computePool2DInfo(input4D.shape, filterSize, strides, pad);
    return this.executeOp("avgPool", () => {
      const res = this.backend.avgPool(input4D, convInfo);
      if (reshapedTo4D) {
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as NDArray as
            T;
      }
      return res as NDArray as T;
    });
  }

  /*
   * Bilinear resize a 3D array per each channel to a new 2D shape.
   * @param x The input Array3D.
   * @param newShape2D The new shape to resize the Array3D to. Each channel is
   * resized individually.
   * @param alignCorners An optional bool. Defaults to False. If true, rescale
   * input by (new_height - 1) / (height - 1), which exactly aligns the 4
   * corners of images and resized images. If false, rescale by new_height /
   * height. Treat similarly the width dimension.
   */
  resizeBilinear3D(
      x: Array3D, newShape2D: [number, number], alignCorners = false): Array3D {
    util.assert(
        x.rank === 3,
        `Error in resizeBilinear3D: x must be rank 3 but got rank ${x.rank}.`);
    util.assert(
        newShape2D.length === 2,
        `Error in resizeBilinear3D: new shape must 2D, but got shape ` +
            `${newShape2D}.`);
    return this.backend.resizeBilinear3D(x, newShape2D, alignCorners);
  }

  //////////////
  // LSTM ops //
  //////////////

  /**
   * Computes the next states and outputs of a stack of LSTMCells.
   * Each cell output is used as input to the next cell.
   * This is only the forward mode.
   * Derived from tf.contrib.rn.MultiRNNCell.
   * @param lstmCells Array of LSTMCell functions.
   * @param data The input to the cell.
   * @param c Array of previous cell states.
   * @param h Array of previous cell outputs.
   * @return Tuple [nextCellStates, cellOutputs]
   */
  multiRNNCell(
      lstmCells: LSTMCell[], data: Array2D, c: Array2D[],
      h: Array2D[]): [Array2D[], Array2D[]] {
    const res = this.scope(() => {
      let input = data;
      const newStates = [];
      for (let i = 0; i < lstmCells.length; i++) {
        const output = lstmCells[i](input, c[i], h[i]);
        newStates.push(output[0]);
        newStates.push(output[1]);
        input = output[1];
      }

      return newStates;
    });
    const newC: Array2D[] = [];
    const newH: Array2D[] = [];
    for (let i = 0; i < res.length; i += 2) {
      newC.push(res[i]);
      newH.push(res[i + 1]);
    }
    return [newC, newH];
  }

  /**
   * Computes the next state and output of a BasicLSTMCell.
   * This is only the forward mode.
   * Derived from tf.contrib.rnn.BasicLSTMCell.
   * @param forgetBias Forget bias for the cell.
   * @param lstmKernel The weights for the cell.
   * @param lstmBias The bias for the cell.
   * @param data The input to the cell.
   * @param c Previous cell state.
   * @param h Previous cell output.
   * @return Tuple [nextCellState, cellOutput]
   */
  basicLSTMCell(
      forgetBias: Scalar, lstmKernel: Array2D, lstmBias: Array1D, data: Array2D,
      c: Array2D, h: Array2D): [Array2D, Array2D] {
    const res = this.scope(() => {
      const combined = this.concat2D(data, h, 1);
      const weighted = this.matMul(combined, lstmKernel);
      const res = this.add(weighted, lstmBias) as Array2D;

      // i = input_gate, j = new_input, f = forget_gate, o = output_gate
      const batchSize = res.shape[0];
      const sliceCols = res.shape[1] / 4;
      const sliceSize: [number, number] = [batchSize, sliceCols];
      const i = this.slice2D(res, [0, 0], sliceSize);
      const j = this.slice2D(res, [0, sliceCols], sliceSize);
      const f = this.slice2D(res, [0, sliceCols * 2], sliceSize);
      const o = this.slice2D(res, [0, sliceCols * 3], sliceSize);

      const newC = this.addStrict(
          this.multiplyStrict(
              c, this.sigmoid(this.scalarPlusArray(forgetBias, f))),
          this.multiplyStrict(this.sigmoid(i), this.tanh(j)));
      const newH = this.multiplyStrict(this.tanh(newC), this.sigmoid(o));

      return [newC, newH];
    });
    return [res[0], res[1]];
  }

  /**
   * Draws samples from a multinomial distribution.
   *
   * @param probabilities 1D array with normalized outcome probabilities, or
   *     2D array of shape `[batchSize, numOutcomes]`.
   * @param numSamples Number of samples to draw for each row slice.
   * @param seed Optional. The seed number.
   * @return 1D array of shape `[numSamples]`, or 2D array of shape
   *     `[batchSize, numSamples]`, depending on the rank of the input.
   */
  multinomial(
      probabilities: Array1D | Array2D, numSamples: number,
      seed?: number): Array1D<"int32"> | Array2D<"int32"> {
    const numOutcomes = probabilities.size;
    if (numOutcomes < 2) {
      throw new Error(
          `Error in multinomial: you need at least 2 outcomes, but got ` +
          `${numOutcomes}.`);
    }
    if (probabilities.rank > 2) {
      throw new Error(
          `Rank of probabilities must be 1 or 2, but is ${probabilities.rank}`);
    }
    seed = seed || Math.random();
    const origRank = probabilities.rank;

    if (probabilities.rank === 1) {
      probabilities = probabilities.as2D(1, -1);
    }
    return this.executeOp("multinomial", () => {
      const res =
        this.backend.multinomial(probabilities as Array2D, numSamples, seed);
      if (origRank === 1) {
        return res.as1D();
      }
      return res;
    });
  }

  /**
   * Returns a one-hot array. The locations represented by `indices` take
   * value `onValue` (defaults to 1), while all other locations take value
   * `offValue` (defaults to 0).
   *
   * @param indices 1D Array of indices.
   * @param depth The depth of the one hot dimension.
   * @param onValue A number used to fill in output when the index matches the
   *     location.
   * @param offValue A number used to fill in the output when the index does
   *     not match the location.
   */
  oneHot(indices: Array1D, depth: number, onValue = 1, offValue = 0): Array2D {
    if (depth < 2) {
      throw new Error(`Error in oneHot: depth must be >=2, but it is ${depth}`);
    }
    return this.backend.oneHot(indices, depth, onValue, offValue);
  }

  /**
   * Calculates the mean and variance of `x`. The mean and variance are
   * calculated by aggregating the contents of `x` across `axes`. If `x` is
   * 1-D and `axes = [0]` this is just the mean and variance of a vector.
   *
   * @param x The input array.
   * @param axis Optional. The dimension(s) along with to compute mean and
   *     variance. By default it reduces all dimensions.
   * @param keepDims If true, the moments have the same dimensionality as the
   *     input.
   * @return An object with two keys: `mean` and `variance`.
   */
  moments(x: NDArray, axis: number | number[] = null, keepDims = false):
      {mean: NDArray<"float32">, variance: NDArray<"float32">} {
    const axes = axis_util.parseAxisParam(axis, x.shape);
    const result = this.scope(() => {
      const mean = this.mean(x, axes, keepDims);
      let keepDimsShape = mean.shape;
      if (!keepDims) {
        keepDimsShape = axis_util.expandShapeToKeepDim(mean.shape, axes);
      }
      const devSquared = this.square(
          this.subtract(x.asType("float32"), mean.reshape(keepDimsShape)));
      const variance = this.mean(devSquared, axes, keepDims);
      return {mean, variance};
    });
    return result;
  }

  /**
   * Computes the norm of scalar, vectors, and matrices.
   * This function can compute several different vector norms (the 1-norm, the
   * Euclidean or 2-norm, the inf-norm, and in general the p-norm for p > 0)
   * and matrix norms (Frobenius, 1-norm, and inf-norm).
   *
   * @param x The input array.
   * @param ord Optional. Order of the norm. Supported norm types are
   * following: ord         norm for matrices          norm for vectors
   *     -------------------------------------------------------
   *     'euclidean' Frobenius norm             2-norm
   *     ‘fro’       Frobenius norm	            –
   *     Infinity    max(sum(abs(x), axis=1))   max(abs(x))
   *     -Infinity   min(sum(abs(x), axis=1))   min(abs(x))
   *     1           max(sum(abs(x), axis=0))   sum(abs(x))
   *     2           -                          sum(abs(x)^2)^1/2*
   *
   * @param axis Optional. If axis is null (the default), the input is
   * considered a vector and a single vector norm is computed over the entire
   * set of values in the NDArray, i.e. norm(x, ord) is equivalent
   * to norm(x.reshape([-1]), ord). If axis is a integer, the input
   * is considered a batch of vectors, and axis determines the axis in x
   * over which to compute vector norms. If axis is a 2-tuple of integer it is
   * considered a batch of matrices and axis determines the axes in NDArray
   * over which to compute a matrix norm.
   * @param keepDims Optional. If true, the norm have the same dimensionality
   * as the input.
   */
  norm<D extends DataType>(
      x: NDArray<D>, ord: number | "euclidean" | "fro" = "euclidean",
      axis: number | number[] = null, keepDims = false):
      NDArray<D | SumTypes[D]> {
    return this.scope(() => {
      const norm = this.normInternal(x, ord, axis);
      let keepDimsShape = norm.shape;
      if (keepDims) {
        const axes = axis_util.parseAxisParam(axis, x.shape);
        keepDimsShape = axis_util.expandShapeToKeepDim(norm.shape, axes);
      }
      return norm.reshape(keepDimsShape);
    });
  }

  setDiag(input: Array2D, diag: Array1D): Array2D {
    return this.backend.setDiag(input, diag);
  }

  /**
   * Calculate the norm for different NDAarray.
   */
  private normInternal<D extends DataType>(
      x: NDArray<D>, p: number | string,
      axis: number | number[] = null): NDArray<D | SumTypes[D]> {
    // scalar
    if (x.rank === 0) {
      return this.abs(x);
    }

    // consider vector when no axis is specified
    if (x.rank !== 1 && axis === null) {
      return this.normInternal(x.reshape([-1]), p, axis);
    }

    // vector
    if (x.rank === 1 || typeof axis === "number" ||
        axis instanceof Array && axis.length === 1) {
      if (p === 1) {
        return this.sum(this.abs(x), axis);
      }
      if (p === Infinity) {
        return this.max(this.abs(x), axis);
      }
      if (p === -Infinity) {
        return this.min(this.abs(x), axis);
      }
      if (p === "euclidean" || p === 2) {
        // norm(x, 2) = sum(abs(xi) ^ 2) ^ 1/2
        return this.sqrt(
            this.sum(this.pow(this.abs(x), Scalar.new(2, "int32")), axis));
      }

      throw new Error(`Error in norm: invalid ord value: ${p}`);
    }

    // matrix (assumption axis[0] < axis[1])
    if (axis instanceof Array && axis.length === 2) {
      if (p === 1) {
        return this.max(this.sum(this.abs(x), axis[0]), axis[1] - 1);
      }
      if (p === Infinity) {
        return this.max(this.sum(this.abs(x), axis[1]), axis[0]);
      }
      if (p === -Infinity) {
        return this.min(this.sum(this.abs(x), axis[1]), axis[0]);
      }
      if (p === "fro" || p === "euclidean") {
        // norm(x) = sqrt(sum(pow(x, 2)))
        return this.sqrt(this.sum(this.pow(x, Scalar.new(2, "int32")), axis));
      }

      throw new Error(`Error in norm: invalid ord value: ${p}`);
    }

    throw new Error(`Error in norm: invalid axis: ${axis}`);
  }

  disposeData(dataId: DataId): void {
    if (!this.registeredArrays.has(dataId)) {
      return;
    }
    const refCount = this.registeredArrays.get(dataId);
    if (refCount <= 1) {
      this.registeredArrays.delete(dataId);
      this.backend.disposeData(dataId);
      this.registeredArrayCount--;
    } else {
      this.registeredArrays.set(dataId, refCount - 1);
    }
    // TODO(nsthorat): Construct an error and save the stack trace for
    // debugging when in debug mode. Creating a stack trace is too expensive
    // to do unconditionally.
  }
}

function parseTupleParam(param: number | [number, number]): [number, number] {
  return typeof param === "number" ? [param, param] : param;
}
