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

import * as test_util from "../../test_util";
import { Tests } from "../../test_util";
import { DataId } from "../ndarray";
import { MathBackendWebGL } from "./backend_webgl";

const tests: Tests = () => {
  it("reading", () => {
    const backend = new MathBackendWebGL(null);
    const texManager = backend.getTextureManager();
    const id = new DataId();
    backend.register(id, [3], "float32");
    backend.write(id, new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(0);
    backend.getTexture(id);
    expect(texManager.getNumUsedTextures()).toBe(1);
    test_util.expectArraysClose(
        backend.readSync(id), new Float32Array([1, 2, 3]));
    expect(texManager.getNumUsedTextures()).toBe(0);
    backend.getTexture(id);
    expect(texManager.getNumUsedTextures()).toBe(1);
    backend.disposeData(id);
    expect(texManager.getNumUsedTextures()).toBe(0);
  });

  it("overwriting", () => {
    const backend = new MathBackendWebGL(null);
    const texManager = backend.getTextureManager();
    const id = new DataId();
    backend.register(id, [3], "float32");
    backend.write(id, new Float32Array([1, 2, 3]));
    backend.getTexture(id);
    expect(texManager.getNumUsedTextures()).toBe(1);
    // overwrite.
    backend.write(id, new Float32Array([4, 5, 6]));
    expect(texManager.getNumUsedTextures()).toBe(0);
    test_util.expectArraysClose(
        backend.readSync(id), new Float32Array([4, 5, 6]));
    backend.getTexture(id);
    expect(texManager.getNumUsedTextures()).toBe(1);
    test_util.expectArraysClose(
        backend.readSync(id), new Float32Array([4, 5, 6]));
    expect(texManager.getNumUsedTextures()).toBe(0);
  });

   it("disposal of backend disposes all textures", () => {
    const backend = new MathBackendWebGL(null);
    const texManager = backend.getTextureManager();
    const id1 = new DataId();
    const id2 = new DataId();
    backend.register(id1, [3], "float32");
    backend.write(id1, new Float32Array([1, 2, 3]));
    backend.getTexture(id1); // Forces upload to GPU.
    backend.register(id2, [3], "float32");
    backend.write(id2, new Float32Array([4, 5, 6]));
    backend.getTexture(id2); // Forces upload to GPU.
    expect(texManager.getNumUsedTextures()).toBe(2);
    backend.dispose();
    expect(texManager.getNumUsedTextures()).toBe(0);
  });
};

test_util.describeCustom("backend_webgl", tests, [
  {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
  {"WEBGL_VERSION": 2, "WEBGL_FLOAT_TEXTURE_ENABLED": true},
  {"WEBGL_VERSION": 1, "WEBGL_FLOAT_TEXTURE_ENABLED": false}
]);
