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

// tslint:disable-next-line:max-line-length
import {NDArrayMathGPU, Scalar} from 'deeplearn';

async function onePlusOne() {
  const math = new NDArrayMathGPU();
  const a = Scalar.new(1);
  const b = Scalar.new(1);

  const result = await math.add(a, b).data();

  document.getElementById('output').innerText = result.toString();
}

onePlusOne();
