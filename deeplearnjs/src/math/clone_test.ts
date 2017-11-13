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

import * as test_util from '../test_util';
import {MathTests} from '../test_util';

import {Array2D} from './ndarray';

const commonTests: MathTests = it => {
  it('returns a ndarray with the same shape and value', math => {
    const a = Array2D.new([3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const aPrime = math.clone(a);
    expect(aPrime.shape).toEqual(a.shape);
    test_util.expectArraysClose(aPrime.getValues(), a.getValues());
    a.dispose();
  });
};

const gpuTests: MathTests = it => {
  it('returns a ndarray with a different texture handle', math => {
    const a = Array2D.new([3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const aPrime = math.clone(a);
    expect(a.inGPU()).toEqual(true);
    expect(aPrime.inGPU()).toEqual(true);
    expect(aPrime.getTexture()).not.toBe(a.getTexture());
    a.dispose();
  });
};

test_util.describeMathCPU('clone', [commonTests]);
test_util.describeMathGPU('clone', [commonTests, gpuTests], [
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
]);
