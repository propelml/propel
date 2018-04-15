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

export class SelectProgram implements GPGPUProgram {
  variableNames = ["Cond", "A", "B"];
  outputShape: number[] = [];
  userCode: string;

  constructor(shape: number[]) {
    this.outputShape = shape.slice();
    const dType = getCoordsDataType(shape.length);
    const sampleCoords = getSampleCoords(shape.length);

    this.userCode = `
      void main() {
        ${dType} coords = getOutputCoords();
        int cond = int(getCond(${sampleCoords}));
        float value;
        if (cond != 0) {
          value = getA(${sampleCoords});
        } else {
          value = getB(${sampleCoords});
        }
        setOutput(value);
      }
    `;
  }
}

function getSampleCoords(rank: number): string {
  if (rank === 1) {
    return "coords";
  }
  if (rank > 4) {
    throw Error(`Select rank ${rank} is not yet supported`);
  }
  const v = ["coords.x", "coords.y", "coords.z", "coords.w"];
  return v.slice(0, rank).join(", ");
}
