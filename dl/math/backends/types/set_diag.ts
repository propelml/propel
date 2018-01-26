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
import {Array1D, Array2D} from '../../ndarray';
// tslint:disable-next-line:max-line-length
import {KernelInputConfig, KernelNode} from '../tape_types';

export interface SetDiagNode extends KernelNode {
  inputAndArgs: SetDiagInputConfig;
  output: Array2D;
}

export interface SetDiagInputConfig extends KernelInputConfig {
  inputs: SetDiagInputArrays;
}

export interface SetDiagInputArrays extends NamedArrayMap {
  input: Array2D;
  diag: Array1D;
}
