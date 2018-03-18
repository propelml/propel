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

export class PadProgram implements GPGPUProgram {
  variableNames = ["X"];
  outputShape: number[] = [];
  userCode: string;

  constructor(shape: number[], paddings: Array<[number, number]>,
              padValue: number) {
    this.outputShape = shape.slice();
    const off = [];
    for (let d = 0; d < shape.length; d++) {
      this.outputShape[d] += paddings[d][0] + paddings[d][1];
      off.push(paddings[d][0]);
    }
    const dType = getCoordsDataType(shape.length);
    let coords = ["coords.x", "coords.y", "coords.z", "coords.w"]
      .slice(0, shape.length);
    if (coords.length === 1) {
      coords = ["coords"];
    }
    const cond = coords.map((coord, i) =>
      `${coord} < ${off[i]} || ${coord} >= ${off[i]} + ${shape[i]}`)
      .join(" || ");
    const getXParams = coords.map((coord, i) => `${coord} - ${off[i]}`)
      .join(", ");

    this.userCode = `
      void main() {
        ${dType} coords = getOutputCoords();
        float value;
        if (${cond}) {
          value = float(${padValue});
        } else {
          value = getX(${getXParams});
        }
        setOutput(value);
      }
    `;
  }
}
