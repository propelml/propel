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

 // TODO src/experiements.ts also has a sgd method - which calls into
 // this module. It's undesirable to have two entry points like that.
 // this module should be favored over the old exp.sgd().

import { gradParams } from "./backprop";
import { Params, params as createParams } from "./params";
import { gc, NamedTensors, Tensor } from "./tensor";
import { assert } from "./util";

export interface SGDOpts {
  lr: number;
  momentum?: number; // TODO currently unused.
  params?: Params;
}

export type LossFn = (params: Params) => Tensor;

// Optimizer is expected to modify the params in someway.
export type Optimizer = (opts, params: Params, grads: NamedTensors) => void;

interface MinimizeResult {
  loss: Tensor;
}

/** Performs SGD given the loss and current parameters. */
export function sgd(opts: SGDOpts, lossFn: LossFn): MinimizeResult {
  return minimize(optimizerSGD, opts, lossFn);
}

export function optimizerSGD(opts, params: Params, grads: NamedTensors): void {
  for (const name of Object.keys(grads)) {
    const g = grads[name];
    const p = params.get(name);
    // p -= g * lr
    p.assign(p.sub(g.mul(opts.lr)));
  }
  // TODO return grads.
}

/** Modifies the current parameters given the loss and optimizer. */
export function minimize(optimizer: Optimizer,
                         opts: SGDOpts,
                         lossFn: LossFn): MinimizeResult {
  const params = opts.params || createParams();
  let loss;
  gc((keep) => {
    const gradFn = gradParams(lossFn);
    // Forward/Backward pass
    params.isTraining = true;
    const gradsAndLoss = gradFn(params);
    params.isTraining = false;
    const grads = gradsAndLoss[0];
    loss = gradsAndLoss[1];
    assert(loss.rank === 0);
    keep(loss);
    optimizer(opts, params, grads);
  });
  return { loss };
}
