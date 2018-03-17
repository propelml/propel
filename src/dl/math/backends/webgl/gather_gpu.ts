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

import { GPGPUProgram } from "./gpgpu_math";
import { getCoordsDataType } from "./shader_compiler";

export class GatherProgram implements GPGPUProgram {
  variableNames = ["X", "Indices"];
  outputShape: number[] = [];
  userCode: string;

  constructor(shape: number[], nIndices: number, axis: number) {
    this.outputShape = shape.slice();
    this.outputShape[axis] = nIndices;
    const dType = getCoordsDataType(shape.length);
    const sampleCoords = getSampleCoords(axis, shape.length);

    this.userCode = `
      void main() {
        ${dType} coords = getOutputCoords();
        float value = getX(${sampleCoords});
        setOutput(value);
      }
    `;
  }
}

function getSampleCoords(axis: number, rank: number): string {
  if (rank > 4 || axis < 0 || axis >= rank) {
    throw Error(`Gather for rank ${rank} is not yet supported`);
  }
  const v =  ["coords.x", "coords.y", "coords.z", "coords.w"];
  v[axis] = "int(getIndices(" + v[axis] + "))";
  return v.slice(0, rank).join(", ");
}
