// Exports of this file is the only public API.
import {NDArrayMath} from './deeplearnjs/src/math/math'
import {Tensor, TensorLike} from './tensor';
import * as backprop from './backprop';

function sp(x: TensorLike): Tensor {
  return Tensor.convert(x);
}

export default sp;

namespace sp {
  export const grad = backprop.grad;
  export const multigrad = backprop.multigrad;
}

