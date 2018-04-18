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

import { Conv2DInfo } from "../conv_util";
import { Array1D, Array2D, Array3D, Array4D, DataId, DataType, DataTypeMap,
  IntDType, NDArray, Rank } from "../ndarray";
import { MatrixOrientation, SumTypes } from "../types";

export interface NDArrayStorage {
  read<D extends DataType>(dataId: DataId): Promise<DataTypeMap[D]>;
  readSync<D extends DataType>(dataId: DataId): DataTypeMap[D];
  disposeData(dataId: DataId): void;
  write<D extends DataType>(dataId: DataId, values: DataTypeMap[D]): void;
  writePixels(
      dataId: DataId,
      pixels: ImageData | HTMLImageElement | HTMLCanvasElement |
              HTMLVideoElement,
      numChannels: number): void;
  time(query: () => NDArray): Promise<number>;
  register(dataId: DataId, shape: number[], dtype: DataType): void;
}

/**
 * The interface that defines the kernels that should be implemented when
 * adding a new backend. New backends don't need to implement every one of the
 * methods, this can be done gradually (throw an error for unimplemented
 * methods).
 */
export interface MathBackend extends NDArrayStorage {
  matMul(
      a: Array2D, b: Array2D, aOrientation: MatrixOrientation,
      bOrientation: MatrixOrientation): Array2D;

  clone<T extends NDArray>(ndarray: T): T;

  gather(x: NDArray, indices: Array1D<"int32">, axis: number): NDArray;
  pad(x: NDArray, paddings: Array<[number, number]>,
      padValue: number): NDArray;

  slice1D(x: Array1D, begin: number, size: number): Array1D;
  slice2D(x: Array2D, begin: [number, number], size: [number, number]): Array2D;
  slice3D(x: Array3D, begin: [number, number, number], size: [
    number, number, number
  ]): Array3D;
  slice4D(x: Array4D, begin: [number, number, number, number], size: [
    number, number, number, number
  ]): Array4D;

  concat1D(a: Array1D, b: Array1D): Array1D;
  concat2D(a: Array2D, b: Array2D, axis: number): Array2D;
  concat3D(a: Array3D, b: Array3D, axis: number): Array3D;
  concat4D(a: Array4D, b: Array4D, axis: number): Array4D;

  neg<T extends NDArray>(a: T): T;

  add<D extends DataType>(a: NDArray<D>, b: NDArray<D>): NDArray<D>;
  subtract<D extends DataType>(a: NDArray<D>, b: NDArray<D>): NDArray<D>;
  multiply<D extends DataType>(a: NDArray<D>, b: NDArray<D>): NDArray<D>;
  divide<D extends DataType>(a: NDArray<D>, b: NDArray<D>): NDArray<"float32">;

  sum<D extends DataType>(x: NDArray<D>, axes: number[]): NDArray<SumTypes[D]>;

  argMin(x: NDArray, axes: number[]): NDArray<"int32">;
  argMax(x: NDArray, axes: number[]): NDArray<"int32">;

  equal(a: NDArray, b: NDArray): NDArray<"bool">;
  notEqual(a: NDArray, b: NDArray): NDArray<"bool">;
  greater(a: NDArray, b: NDArray): NDArray<"bool">;
  greaterEqual(a: NDArray, b: NDArray): NDArray<"bool">;
  less(a: NDArray, b: NDArray): NDArray<"bool">;
  lessEqual(a: NDArray, b: NDArray): NDArray<"bool">;
  select(cond: NDArray<"bool">, a: NDArray, b: NDArray): NDArray;

  topKValues<D extends DataType, T extends NDArray<D>>(x: T, k: number):
      Array1D<D>;
  topKIndices(x: NDArray, k: number): Array1D<"int32">;

  min<D extends DataType>(x: NDArray<D>, axes: number[]): NDArray<D>;
  minimum<D extends DataType>(a: NDArray<D>, b: NDArray<D>): NDArray<D>;

  max<D extends DataType>(x: NDArray<D>, axes: number[]): NDArray<D>;
  maximum<D extends DataType>(a: NDArray<D>, b: NDArray<D>): NDArray<D>;

  ceil<T extends NDArray>(x: T): T;
  floor<T extends NDArray>(x: T): T;

  pow<T extends NDArray>(a: T, b: NDArray): T;
  exp<T extends NDArray>(x: T): T;
  log<T extends NDArray>(x: T): T;
  sqrt<T extends NDArray>(x: T): T;

  square<T extends NDArray>(x: T): T;

  relu<T extends NDArray>(x: T): T;
  elu<T extends NDArray>(x: T): T;
  eluDer<T extends NDArray>(x: T): T;
  selu<T extends NDArray>(x: T): T;
  leakyRelu<T extends NDArray>(x: T, alpha: number): T;
  prelu<T extends NDArray>(x: T, alpha: T): T;
  preluDer<T extends NDArray>(x: T, alpha: T): T;
  int<D extends IntDType, R extends Rank>(
      x: NDArray<DataType, R>, dtype: D): NDArray<D, R>;

  clip<T extends NDArray>(x: T, min: number, max: number): T;

  abs<T extends NDArray>(x: T): T;

  sigmoid<T extends NDArray>(x: T): T;

  sin<T extends NDArray>(x: T): T;
  cos<T extends NDArray>(x: T): T;
  tan<T extends NDArray>(x: T): T;

  asin<T extends NDArray>(x: T): T;
  acos<T extends NDArray>(x: T): T;
  atan<T extends NDArray>(x: T): T;

  sinh<T extends NDArray>(x: T): T;
  cosh<T extends NDArray>(x: T): T;
  tanh<T extends NDArray>(x: T): T;

  step<T extends NDArray>(x: T, alpha: number): T;

  conv2d(x: Array4D, filter: Array4D, bias: Array1D | null,
         convInfo: Conv2DInfo): Array4D;
  conv2dDerInput(dy: Array4D, filter: Array4D, convInfo: Conv2DInfo): Array4D;
  conv2dDerFilter(x: Array4D, dY: Array4D, convInfo: Conv2DInfo): Array4D;
  conv2dDerBias(dY: Array4D): Array1D;

  depthwiseConv2D(input: Array4D, filter: Array4D, convInfo: Conv2DInfo):
      Array4D;

  maxPool(x: Array4D, convInfo: Conv2DInfo): Array4D;
  maxPoolBackprop(dy: Array4D, x: Array4D, convInfo: Conv2DInfo): Array4D;

  minPool(x: Array4D, convInfo: Conv2DInfo): Array4D;
  avgPool(x: Array4D, convInfo: Conv2DInfo): Array4D;

  tile<D extends DataType, T extends NDArray<D>>(x: T, reps: number[]): T;

  transpose<D extends DataType, T extends NDArray<D>>(x: T, perm: number[]): T;

  resizeBilinear3D(
      x: Array3D, newShape2D: [number, number], alignCorners: boolean): Array3D;

  multinomial(probabilities: Array2D, numSamples: number, seed: number):
      Array2D<"int32">;

  oneHot(indices: Array1D, depth: number, onValue: number, offValue: number):
      Array2D;

  setDiag(input: Array2D, diag: Array1D): Array2D;

  unsortedSegmentSum(data: NDArray, segmentIds: Array1D<"int32">,
                     numSegments: number): NDArray;

  dispose(): void;
}
