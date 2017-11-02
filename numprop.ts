// Exports of this file is the only public API.
import {NDArrayMath} from './deeplearnjs/src/math/math'
import {Tensor, TensorLike} from './tensor';
export {Tensor, TensorLike} from './tensor';
import * as backprop from './backprop';
import * as ops from './ops';
export * from './util';

export let grad = backprop.grad;

export function exp(a: TensorLike): Tensor {
  a = Tensor.convert(a);
  return (new ops.Exp()).run(a);
}

export function neg(a: TensorLike): Tensor {
  a = Tensor.convert(a);
  return (new ops.Neg()).run(a);
}

export function add(a: TensorLike, b: TensorLike) {
  a = Tensor.convert(a);
  b = Tensor.convert(b);
  return (new ops.Add()).run(a, b);
}

export function sub(a: TensorLike, b: TensorLike) {
  a = Tensor.convert(a);
  b = Tensor.convert(b);
  return (new ops.Sub()).run(a, b);
}

export function div(a: TensorLike, b: TensorLike) {
  a = Tensor.convert(a);
  b = Tensor.convert(b);
  return (new ops.Div()).run(a, b);
}

export function mul(a: TensorLike, b: TensorLike) {
  a = Tensor.convert(a);
  b = Tensor.convert(b);
  return (new ops.Mul()).run(a, b);
}
