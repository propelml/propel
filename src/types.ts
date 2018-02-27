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
export type Shape = number[];
export type DType = "float32" | "int32" | "uint8" | "bool";
export type TypedArray = Float32Array | Int32Array | Uint8Array;
export type FlatVector = number[] | TypedArray;
export type RegularArray<T> = T[] | T[][] | T[][][] | T[][][][];
export type ShapeDType = [Shape, DType];
export type ShapeDTypeList = Array<null | ShapeDType>;
// JavaScript objects that can be generally converted to Tensors.
export type Convertible = number | RegularArray<number> | TypedArray;
export type TensorLike = BasicTensor | Convertible;
export type DeviceType = "CPU" | "GPU";

// BasicTensor does not use backprop.
// TODO rename BasicTensor to Storage like in pytorch
// http://pytorch.org/docs/0.3.0/storage.html
export interface BasicTensor {
  readonly shape: Shape;
  readonly dtype: DType;
  dataSync(): TypedArray;
  data(): Promise<TypedArray>;
  dispose(): void;
}

// BackendOps do not use backprop.
export interface BackendOps {
  copyToDevice(x: BasicTensor, device: string): BasicTensor;
  getDevice(x: BasicTensor): string;
  listDevices(): string[];
  add(x: BasicTensor, y: BasicTensor): BasicTensor;
  sub(x: BasicTensor, y: BasicTensor): BasicTensor;
  mul(x: BasicTensor, y: BasicTensor): BasicTensor;
  div(x: BasicTensor, y: BasicTensor): BasicTensor;
  neg(x: BasicTensor): BasicTensor;
  exp(x: BasicTensor): BasicTensor;
  log(x: BasicTensor): BasicTensor;
  setDiag(input: BasicTensor, diag: BasicTensor): BasicTensor;
  onesLike(x: BasicTensor): BasicTensor;
  zerosLike(x: BasicTensor): BasicTensor;
  fill(value: BasicTensor, shape: Shape): BasicTensor;
  square(x: BasicTensor): BasicTensor;
  sinh(x: BasicTensor): BasicTensor;
  cosh(x: BasicTensor): BasicTensor;
  tanh(x: BasicTensor): BasicTensor;
  relu(x: BasicTensor): BasicTensor;
  // reluGrad should not be exposed to the public API.
  // Generally Propel wants to express gradient functions in terms of other
  // known ops. However due to the ubiquity and performance necessities of
  // ReLU, we break this design goal and expose a special op for ReLU's
  // backward pass.
  reluGrad(grads: BasicTensor, features: BasicTensor): BasicTensor;
  sigmoid(x: BasicTensor): BasicTensor;
  abs(x: BasicTensor): BasicTensor;
  randn(shape: Shape, seed?: number): BasicTensor;
  linspace(start: number, stop: number, num: number): BasicTensor;
  range(start: number, limit: number, delta: number): BasicTensor;
  transpose(x: BasicTensor, perm: BasicTensor): BasicTensor;
  reverse(x: BasicTensor, dims: BasicTensor): BasicTensor;
  matmul(x: BasicTensor, y: BasicTensor, transposeA: boolean,
         transposeB: boolean): BasicTensor;
  argmax(x: BasicTensor, axis: number): BasicTensor;
  argmin(x: BasicTensor, axis: number): BasicTensor;
  reduceSum(x: BasicTensor, axes: number[], keepDims: boolean): BasicTensor;
  reduceMean(x: BasicTensor, axes: number[], keepDims: boolean): BasicTensor;
  reduceMax(x: BasicTensor, axes: number[], keepDims: boolean): BasicTensor;
  slice(x: BasicTensor, begin: number[], size: number[]): BasicTensor;
  concat(axis: number, inputs: BasicTensor[]): BasicTensor;
  reshape(x: BasicTensor, newShape: Shape): BasicTensor;
  equal(x: BasicTensor, y: BasicTensor): BasicTensor;
  greater(x: BasicTensor, y: BasicTensor): BasicTensor;
  greaterEqual(x: BasicTensor, y: BasicTensor): BasicTensor;
  less(x: BasicTensor, y: BasicTensor): BasicTensor;
  lessEqual(x: BasicTensor, y: BasicTensor): BasicTensor;
  select(cond: BasicTensor, t: BasicTensor, f: BasicTensor): BasicTensor;
  sign(x: BasicTensor): BasicTensor;
  softmax(x: BasicTensor): BasicTensor;
  logSoftmax(x: BasicTensor): BasicTensor;
  cast(x: BasicTensor, dtype: DType): BasicTensor;
  oneHot(x: BasicTensor, depth: number, onValue: number,
         offValue: number): BasicTensor;
}

// A TapeEntry is created every time an op is executed. It is the bookkeeping
// entry for backpropigation.
export interface TapeEntry {
  name: string;
  oid: number;
  inputIds: number[];
  inputShapeDTypes: ShapeDTypeList;
  outputIds: number[];
  savedForBackward: any[];
}

/** TensorOpts are used to build Tensors in functions like tensor() and zeros().
 * Note that Tensors themselves implement the TensorOpts interface, so existing
 * tensors can be used to construct similiarly typed and located tensors.
 */
export interface TensorOpts {
  dtype: DType;
  device?: string;
}
