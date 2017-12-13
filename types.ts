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

// BasicTensor does not use backprop.
export interface BasicTensor {
  readonly shape: Shape;
  readonly dtype: DType;
  getData(): TypedArray;
}

// BackendOps do not use backprop.
export interface BackendOps {
  add(x: BasicTensor, y: BasicTensor): BasicTensor;
  sub(x: BasicTensor, y: BasicTensor): BasicTensor;
  mul(x: BasicTensor, y: BasicTensor): BasicTensor;
  div(x: BasicTensor, y: BasicTensor): BasicTensor;
  neg(x: BasicTensor): BasicTensor;
  exp(x: BasicTensor): BasicTensor;
  log(x: BasicTensor): BasicTensor;
  eye(size: number, dtype: DType): BasicTensor;
  onesLike(x: BasicTensor): BasicTensor;
  zerosLike(x: BasicTensor): BasicTensor;
  fill(value: BasicTensor, shape: Shape): BasicTensor;
  square(x: BasicTensor): BasicTensor;
  sinh(x: BasicTensor): BasicTensor;
  cosh(x: BasicTensor): BasicTensor;
  tanh(x: BasicTensor): BasicTensor;
  randn(shape: Shape, seed?: number): BasicTensor;
  linspace(start: number, stop: number, num: number): BasicTensor;
  arange(start: number, limit: number, delta: number): BasicTensor;
  transpose(x: BasicTensor, perm: BasicTensor): BasicTensor;
  reverse(x: BasicTensor, dims: BasicTensor): BasicTensor;
  matmul(x: BasicTensor, y: BasicTensor, transposeA: boolean,
         transposeB: boolean): BasicTensor;
  argmax(x: BasicTensor, axis: number): BasicTensor;
  argmin(x: BasicTensor, axis: number): BasicTensor;
  reduceSum(x: BasicTensor, axes: number[], keepDims: boolean): BasicTensor;
  reduceMax(x: BasicTensor, axes: number[], keepDims: boolean): BasicTensor;
  slice(x: BasicTensor, begin: number[], size: number[]): BasicTensor;
  reshape(x: BasicTensor, newShape: Shape): BasicTensor;
  equal(x: BasicTensor, y: BasicTensor): BasicTensor;
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

export function isTypedArray(x: any): x is TypedArray {
  return (x instanceof Float32Array || x instanceof Uint8Array ||
          x instanceof Int32Array);
}

export function getDType(data: TypedArray): DType {
  if (data instanceof Int32Array) {
    return "int32";
  } else if (data instanceof Float32Array) {
    return "float32";
  } else if (data instanceof Uint8Array) {
    return "uint8";
  }
}

export function makeTypedArray(data, dtype: DType = "float32"): TypedArray {
  switch (dtype) {
    case "bool":
      return new Uint8Array(data);
    case "float32":
      return new Float32Array(data);
    case "int32":
      return new Int32Array(data);
    case "uint8":
      return new Uint8Array(data);
    default:
      throw new Error("Not implemented");
  }
}
