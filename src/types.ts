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
export type TensorLike = Storage | Convertible;
export type DeviceType = "CPU" | "GPU";
export type Padding = "same" | "valid";
export interface ConvOpts {
  stride?: number | [number, number];
  padding?: Padding;
}
export interface PoolOpts {
  size?: number | [number, number];
  stride?: number | [number, number];
  padding?: Padding;
}

// Storage does not use backprop.
export interface Storage {
  readonly shape: Shape;
  readonly dtype: DType;
  dataSync(): TypedArray;
  data(): Promise<TypedArray>;
  dispose(): void;
}

// BackendOps do not use backprop.
export interface BackendOps {
  copyToDevice(x: Storage, device: string): Storage;
  getDevice(x: Storage): string;
  listDevices(): string[];
  fromTypedArray(data: TypedArray, shape: Shape, dtype?: DType,
                 device?: string): Storage;
  add(x: Storage, y: Storage): Storage;
  sub(x: Storage, y: Storage): Storage;
  mul(x: Storage, y: Storage): Storage;
  div(x: Storage, y: Storage): Storage;
  neg(x: Storage): Storage;
  exp(x: Storage): Storage;
  log(x: Storage): Storage;
  setDiag(input: Storage, diag: Storage): Storage;
  onesLike(x: Storage): Storage;
  zerosLike(x: Storage): Storage;
  fill(value: Storage, shape: Shape): Storage;
  square(x: Storage): Storage;
  pow(x: Storage, exponent: Storage): Storage;
  sqrt(x: Storage): Storage;
  sin(x: Storage): Storage;
  cos(x: Storage): Storage;
  tan(x: Storage): Storage;
  sinh(x: Storage): Storage;
  cosh(x: Storage): Storage;
  tanh(x: Storage): Storage;
  relu(x: Storage): Storage;
  // reluGrad should not be exposed to the public API.
  // Generally Propel wants to express gradient functions in terms of other
  // known ops. However due to the ubiquity and performance necessities of
  // ReLU, we break this design goal and expose a special op for ReLU's
  // backward pass.
  reluGrad(grads: Storage, features: Storage): Storage;
  sigmoid(x: Storage): Storage;
  abs(x: Storage): Storage;
  randn(shape: Shape, seed?: number): Storage;
  linspace(start: number, stop: number, num: number): Storage;
  range(start: number, limit: number, delta: number): Storage;
  transpose(x: Storage, perm: Storage): Storage;
  reverse(x: Storage, dims: Storage): Storage;
  matmul(x: Storage, y: Storage, transposeA: boolean,
         transposeB: boolean): Storage;
  argmax(x: Storage, axis: number): Storage;
  argmin(x: Storage, axis: number): Storage;
  reduceSum(x: Storage, axes: number[], keepDims: boolean): Storage;
  reduceMean(x: Storage, axes: number[], keepDims: boolean): Storage;
  reduceMax(x: Storage, axes: number[], keepDims: boolean): Storage;
  slice(x: Storage, begin: number[], size: number[]): Storage;
  gather(x: Storage, indices: Storage, axis: number): Storage;
  concat(axis: number, inputs: Storage[]): Storage;
  pad(x: Storage, paddings: Array<[number, number]>, padValue: number): Storage;
  reshape(x: Storage, newShape: Shape): Storage;
  equal(x: Storage, y: Storage): Storage;
  greater(x: Storage, y: Storage): Storage;
  greaterEqual(x: Storage, y: Storage): Storage;
  less(x: Storage, y: Storage): Storage;
  lessEqual(x: Storage, y: Storage): Storage;
  select(cond: Storage, t: Storage, f: Storage): Storage;
  sign(x: Storage): Storage;
  softmax(x: Storage): Storage;
  logSoftmax(x: Storage): Storage;
  cast(x: Storage, dtype: DType): Storage;
  oneHot(x: Storage, depth: number, onValue: number,
         offValue: number): Storage;

  conv2d(input: Storage, filter: Storage, opts: ConvOpts): Storage;
  conv2dGradFilter(grad: Storage, input: Storage,
                   filterShape: Shape, opts: ConvOpts): Storage;
  conv2dGradInput(gradient: Storage, inputShape: Shape,
                  filter: Storage, opts: ConvOpts): Storage;
  maxPool(input: Storage, opts: PoolOpts): Storage;
  maxPoolGrad(grad: Storage, origInput: Storage, origOutput: Storage,
              opts: PoolOpts): Storage;
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

export type Mode = "RGBA" | "RGB" | "L";
