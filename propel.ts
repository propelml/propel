/*
   Copyright 2017 propel authors. All Rights Reserved.
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

// Exports of this file is the only public API.

import { Tensor, TensorLike } from './tensor';
import * as backprop from './backprop';
import { assert } from './util';
import * as ops from './ops';

function $(x: TensorLike): Tensor {
  return Tensor.convert(x);
}

export default $;

namespace $ {
  export const grad = backprop.grad;
  export const multigrad = backprop.multigrad;

  // Return evenly spaced numbers over a specified interval.
  //
  // Returns num evenly spaced samples, calculated over the interval
  // [start, stop].
  export const linspace = function(start, stop, num = 50): Tensor {
    let a = [];
    let d = (stop - start) / num;
    for (let i = 0; i < num; ++i) {
      a.push(start + i * d);
    }
    return $(a);
  };

  export const arange = function(start, stop, step = 1): Tensor {
    let a = [];
    for (let i = start; i < stop; i += step) {
      a.push(i);
    }
    return $(a);
  };

  export const tanh = function(x: TensorLike): Tensor {
    let y = $(x).mul(-2).exp();
    return $(1).sub(y).div(y.add(1));
  };

  // Join a sequence of arrays along an existing axis.
  // Current API:
  //   $.concat([a, b, c], 0);  <-- most like np/tf's api
  // Alternate API ideas:
  //   a.concat(b, c, 0);    <-- weird because a, b, c have equal signifigance
  //                             but that can also be said of a.add(b).add(c)
  //                             should allow a.add(b, c)
  //   $(a, b, c).concat(0);
  //   $([a, b, c]).concat(0);
  export const concat = (tensors: TensorLike[], axis = 0) => {
    return ops.concat(axis, ...tensors);
  };

  // Join a sequence of arrays along a new axis.
  //
  // The axis parameter specifies the index of the new axis in the dimensions
  // of the result. For example, if axis=0 it will be the first dimension and
  // if axis=-1 it will be the last dimension.
  // Current API:
  //    $.stack([a, b, c], 0);
  // Alternate API ideas:
  //    a.stack(b, c, 0);
  //    $([a, b, c]).stack(0);
  //    $(a, b, c).stack(0);
  export const stack = function(tensors: TensorLike[], axis = 0): Tensor {
    assert(axis >= 0, "TODO: Negative axis values");
    let tensors_ = tensors.map(t => Tensor.convert(t).expandDims(axis));
    let shapes = tensors_.map(t => t.shape);
    assert($.allEqual(...shapes), "shapes not all equal");
    return ops.concat(axis, ...tensors_);
  };

  export const allEqual = (...tensors: TensorLike[]): boolean => {
    if (tensors.length <= 1) {
      return true // ?
    } else {
      let first = $(tensors[0]);
      for (let i = 1; i < tensors.length; i++) {
        if (!first.equals(tensors[i])) return false;
      }
    }
    return true;
  }
}

