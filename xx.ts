import {NDArrayMath} from './deeplearnjs/src/math/math'
import {NDArray} from './deeplearnjs/src/math/ndarray'
export {NDArray} from './deeplearnjs/src/math/ndarray'
import {NDArrayMathCPU} from './deeplearnjs/src/math/math_cpu';

let math = new NDArrayMathCPU();

function toArray(x: number|NDArray) {
  if (typeof x == "number") {
    return NDArray.make([], {values: new Float32Array([x])});
  }
  return x;
}

export function exp(a) {
  a = toArray(a);
  return math.relu(a);
}

export function neg(a) {
  a = toArray(a);
  return a;
}

export function add(a, b) {
  a = toArray(a);
  b = toArray(b);
  return math.add(a, b);
}

export function sub(a, b) {
  a = toArray(a);
  b = toArray(b);
  return add(a, neg(b));
}

export function div(a, b) {
  a = toArray(a);
  b = toArray(b);
  return math.divide(a, b);
}

export function mul(a, b) {
  a = toArray(a);
  b = toArray(b);
  return math.multiply(a, b);
}

export function grad(a) {
  return a;
}
