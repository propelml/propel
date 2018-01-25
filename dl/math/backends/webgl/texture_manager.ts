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

import {GPGPUContext} from './gpgpu_context';

export class TextureManager {
  private numUsedTextures = 0;

  constructor(private gpgpu: GPGPUContext) {}

  acquireTexture(shapeRC: [number, number]): WebGLTexture {
    this.numUsedTextures++;
    const newTexture = this.gpgpu.createMatrixTexture(shapeRC[0], shapeRC[1]);
    return newTexture;
  }

  releaseTexture(texture: WebGLTexture, shape: [number, number]): void {
    this.numUsedTextures--;
    this.gpgpu.deleteMatrixTexture(texture);
  }

  getNumUsedTextures(): number {
    return this.numUsedTextures;
  }

  dispose() {
  }
}
