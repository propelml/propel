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

import * as environment from "./environment";
import * as gpgpu_util from "./math/backends/webgl/gpgpu_util";
// tslint:disable-next-line:max-line-length
import * as render_ndarray_gpu_util from "./math/backends/webgl/render_ndarray_gpu_util";
import * as webgl_util from "./math/backends/webgl/webgl_util";
import * as conv_util from "./math/conv_util";
import * as test_util from "./test_util";
import * as util from "./util";

export {ENV, Environment, Features} from "./environment";
export {MathBackendCPU} from "./math/backends/backend_cpu";
export {MathBackendWebGL} from "./math/backends/backend_webgl";
export {GPGPUContext} from "./math/backends/webgl/gpgpu_context";
export {LSTMCell, NDArrayMath} from "./math/math";
// tslint:disable-next-line:max-line-length
export {Array1D, Array2D, Array3D, Array4D, NDArray, Scalar} from "./math/ndarray";
export {MatrixOrientation} from "./math/types";
// Second level exports.
export {
  conv_util,
  environment,
  gpgpu_util,
  render_ndarray_gpu_util,
  test_util,
  util,
  webgl_util
};
