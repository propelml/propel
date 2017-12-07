export type Shape = number[];
export type DType = "float32" | "int32" | "uint8" | "bool";
export type TypedArray = Float32Array | Int32Array | Uint8Array;
export type FlatVector = number[] | TypedArray;
export type RegularArray<T> = T[] | T[][] | T[][][] | T[][][][];

// BasicTensor does not use backprop.
export interface BasicTensor {
  readonly shape: Shape;
  readonly dtype: DType;
  getData(): TypedArray;
}

// BasicOps do not use backprop.
export interface BasicOps {
  add(x: BasicTensor, y: BasicTensor): BasicTensor;
  sub(x: BasicTensor, y: BasicTensor): BasicTensor;
  mul(x: BasicTensor, y: BasicTensor): BasicTensor;
  div(x: BasicTensor, y: BasicTensor): BasicTensor;
  neg(x: BasicTensor): BasicTensor;
  exp(x: BasicTensor): BasicTensor;
  eye(size: number, dtype: DType): BasicTensor;
  onesLike(x: BasicTensor): BasicTensor;
  square(x: BasicTensor): BasicTensor;
  sinh(x: BasicTensor): BasicTensor;
  cosh(x: BasicTensor): BasicTensor;
  tanh(x: BasicTensor): BasicTensor;
  randn(shape: Shape, seed?: number): BasicTensor;
  linspace(start: number, stop: number, num: number): BasicTensor;
  arange(start: number, limit: number, delta: number): BasicTensor;
  transpose(x: BasicTensor, perm: BasicTensor): BasicTensor;
  reverse(x: BasicTensor, dims: BasicTensor): BasicTensor;
}

// JavaScript objects that can be generally converted to Tensors.
export type Convertible = number | RegularArray<number> | TypedArray;
export type TensorLike = BasicTensor | Convertible;

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

export function makeTypedArray(data, dtype: DType): TypedArray {
  switch (dtype) {
    case "bool":
      return new Uint8Array(data);
    case "float32":
      return new Float32Array(data);
    case "int32":
      return new Int32Array(data);
    default:
      throw new Error("Not implemented");
  }
}
