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

import * as backprop from "./backprop";
import { Tensor, TensorLike } from "./tensor";

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
    const a = [];
    const d = (stop - start) / num;
    for (let i = 0; i < num; ++i) {
      a.push(start + i * d);
    }
    return Tensor.convert(a);
  };

  export const arange = function(start, stop, step = 1): Tensor {
    const a = [];
    for (let i = start; i < stop; i += step) {
      a.push(i);
    }
    return Tensor.convert(a);
  };

  export const tanh = function(x: TensorLike): Tensor {
    const y = $(x).mul(-2).exp();
    return $(1).sub(y).div(y.add(1));
  };
}
