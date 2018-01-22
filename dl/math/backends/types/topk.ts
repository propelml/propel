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
import {Array1D, DataType, NDArray} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputConfig, KernelNode} from '../tape_types';

// Values
export interface TopKValuesNode<D extends DataType, T extends NDArray<D>>
    extends KernelNode {
  inputAndArgs: TopKValuesInputConfig<T>;
  output: Array1D<D>;
}

export interface TopKValuesInputConfig<T extends NDArray> extends
    KernelInputConfig {
  inputs: TopKValuesInputArrays<T>;
  args: {k: number};
}

export interface TopKValuesInputArrays<T extends NDArray> extends
    NamedArrayMap {
  x: T;
}

// Indices
export interface TopKIndicesNode extends KernelNode {
  inputAndArgs: TopKIndicesInputConfig;
  output: Array1D<'int32'>;
}

export interface TopKIndicesInputConfig extends KernelInputConfig {
  inputs: TopKIndicesInputArrays;
  args: {k: number};
}

export interface TopKIndicesInputArrays extends NamedArrayMap { x: NDArray; }
