/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {NamedArrayMap} from '../../../util';
import {DataType, NDArray} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputConfig, KernelNode} from '../tape_types';

// Reduction min.
export interface MinNode<D extends DataType> extends KernelNode {
  inputAndArgs: MinInputConfig<D>;
  output: NDArray<D>;
}

// Element-wise min.
export interface MinimumNode<D extends DataType> extends KernelNode {
  inputAndArgs: MinimumInputConfig<D>;
  output: NDArray<D>;
}

export interface MinimumInputConfig<D extends DataType> extends
    KernelInputConfig {
  inputs: {a: NDArray<D>, b: NDArray<D>};
}

export interface MinInputConfig<D extends DataType> extends KernelInputConfig {
  inputs: MinInputArrays<D>;
}

export interface MinInputArrays<D extends DataType> extends NamedArrayMap {
  x: NDArray<D>;
}

// Reduction Max
export interface MaxNode<D extends DataType> extends KernelNode {
  inputAndArgs: MaxInputConfig<D>;
  output: NDArray<D>;
}

// Element-wise max.
export interface MaximumNode<D extends DataType> extends KernelNode {
  inputAndArgs: MaximumInputConfig<D>;
  output: NDArray<D>;
}

export interface MaximumInputConfig<D extends DataType> extends
    KernelInputConfig {
  inputs: {a: NDArray<D>, b: NDArray<D>};
}

export interface MaxInputConfig<D extends DataType> extends KernelInputConfig {
  inputs: MaxInputArrays<D>;
}

export interface MaxInputArrays<D extends DataType> extends NamedArrayMap {
  x: NDArray<D>;
}
