import {NDArrayMath} from './deeplearnjs/src/math/math'
import {NDArray} from './deeplearnjs/src/math/ndarray'
export {NDArray} from './deeplearnjs/src/math/ndarray'
import {NDArrayMathCPU} from './deeplearnjs/src/math/math_cpu';

let math = new NDArrayMathCPU();

function toNDArray(x: number|NDArray) {
  if (typeof x == "number") {
    return NDArray.make([], {values: new Float32Array([x])});
  }
  return x;
}

export function assert(expr: boolean, msg: string) {
  if (!expr) {
    throw new Error(msg);
  }
}

export function assertClose(a: number, b: number, msg: string) {
  assert(Math.abs(a - b) < 0.001, msg);
}

export function exp(a) {
  a = toNDArray(a);
  return math.exp(a);
}

export function neg(a) {
  a = toNDArray(a);
  return math.neg(a);
}

export function add(a, b) {
  a = toNDArray(a);
  b = toNDArray(b);
  return math.add(a, b);
}

export function sub(a, b) {
  a = toNDArray(a);
  b = toNDArray(b);
  return add(a, neg(b));
}

export function div(a, b) {
  a = toNDArray(a);
  b = toNDArray(b);
  return math.divide(a, b);
}

export function mul(a, b) {
  a = toNDArray(a);
  b = toNDArray(b);
  return math.multiply(a, b);
}

export function grad(a) {
  return a;
}
