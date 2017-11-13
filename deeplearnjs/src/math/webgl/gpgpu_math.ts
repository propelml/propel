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

import {ENV} from '../../environment';
import * as util from '../../util';
import {NDArray} from '../ndarray';

import {GPGPUContext} from './gpgpu_context';
import * as shader_compiler from './shader_compiler';
import {ShapeInfo} from './shader_compiler';

const ATTRIBUTE_NAMES = ['uv', 'clipSpacePos'];

export interface GPGPUProgram {
  variableNames: string[];
  outputShape: number[];
  userCode: string;
  supportsBroadcasting?: boolean;
  numBatchDims?: number;
}

export interface GPGPUBinary {
  webGLProgram: WebGLProgram;
  program: GPGPUProgram;
  uniformLocations: {[name: string]: WebGLUniformLocation};
  attributeLocations: {[name: string]: number};
  gpgpu: GPGPUContext;
  source: string;
  inShapeInfos: ShapeInfo[];
  outShapeInfo: ShapeInfo;
}

const NAN_UNIFORM_NAME = 'NaN';

function shouldUploadNaNUniform(): boolean {
  return !ENV.get('WEBGL_FLOAT_TEXTURE_ENABLED');
}

export function compileProgram<T extends NDArray, K extends NDArray>(
    gpgpu: GPGPUContext, program: GPGPUProgram, inputs: T[],
    output: K): GPGPUBinary {
  const userCode = program.userCode;
  const inputInfos = inputs.map((input, i) => {
    const shapeInfo = {
      logicalShape: input.shape,
      texShape: input.getTextureShapeRC(),
      textureType: input.getData().textureType
    };
    return {name: program.variableNames[i], shapeInfo};
  });
  const inShapeInfos = inputInfos.map(x => x.shapeInfo);
  const outShapeInfo = {
    logicalShape: output.shape,
    texShape: output.getTextureShapeRC(),
    textureType: output.getData().textureType
  };
  const source = shader_compiler.makeShader(
      inputInfos, outShapeInfo, userCode, program.supportsBroadcasting === true,
      program.numBatchDims);

  const webGLProgram = gpgpu.createProgram(source);

  const uniformLocations: {[name: string]: WebGLUniformLocation} = {};
  for (let i = 0; i < program.variableNames.length; i++) {
    const uniformName = program.variableNames[i];
    uniformLocations[uniformName] =
        gpgpu.getUniformLocation(webGLProgram, uniformName);
  }
  const attributeLocations: {[name: string]: number} = {};
  ATTRIBUTE_NAMES.forEach(attribute => {
    attributeLocations[attribute] =
        gpgpu.getAttributeLocation(webGLProgram, attribute);
  });

  if (shouldUploadNaNUniform()) {
    uniformLocations[NAN_UNIFORM_NAME] =
        gpgpu.getUniformLocation(webGLProgram, NAN_UNIFORM_NAME);
  }

  return {
    program,
    source,
    webGLProgram,
    uniformLocations,
    attributeLocations,
    gpgpu,
    inShapeInfos,
    outShapeInfo
  };
}

function validateBinaryAndProgram(
    shapeInfos: ShapeInfo[], inputs: NDArray[], numBatchDims: number) {
  if (shapeInfos.length !== inputs.length) {
    throw Error(
        `Binary was compiled with ${shapeInfos.length} inputs, but ` +
        `was executed with ${inputs.length} inputs`);
  }

  shapeInfos.forEach((s, i) => {
    const shapeA = s.logicalShape;
    const texShapeA = s.texShape;
    const shapeB = inputs[i].shape;
    const texShapeB = inputs[i].getTextureShapeRC();

    if (!numBatchDims && !util.arraysEqual(shapeA, shapeB)) {
      throw Error(
          `Binary was compiled with different shapes than ` +
          `the current args. Shapes ${shapeA} and ${shapeB} must match`);
    }
    if (!numBatchDims && !util.arraysEqual(texShapeA, texShapeB)) {
      throw Error(
          `Binary was compiled with different texture shapes than the` +
          ` current args. Shape ${texShapeA} and ${texShapeB} must match`);
    }
  });
}

export function runProgram<T extends NDArray, K extends NDArray>(
    binary: GPGPUBinary, inputs: T[], output: K,
    customSetup?: (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) =>
        void): void {
  validateBinaryAndProgram(
      binary.inShapeInfos, inputs, binary.program.numBatchDims);
  validateBinaryAndProgram(
      [binary.outShapeInfo], [output], binary.program.numBatchDims);

  const outTex = output.getTexture();
  const outTexShape = output.getTextureShapeRC();
  const gpgpu = binary.gpgpu;
  gpgpu.setOutputMatrixTexture(outTex, outTexShape[0], outTexShape[1]);
  gpgpu.setProgram(binary.webGLProgram);
  inputs.forEach((input, i) => {
    const tex = input.getTexture();
    const variableName = binary.program.variableNames[i];
    const variableUniformLocation = binary.uniformLocations[variableName];
    gpgpu.setInputMatrixTexture(tex, variableUniformLocation, i);
  });

  if (shouldUploadNaNUniform()) {
    gpgpu.gl.uniform1f(binary.uniformLocations[NAN_UNIFORM_NAME], NaN);
  }

  if (customSetup != null) {
    customSetup(gpgpu, binary.webGLProgram);
  }
  gpgpu.executeProgram(binary.attributeLocations);
}

export function makeShaderKey(
    program: GPGPUProgram, inputs: NDArray[], output: NDArray): string {
  let keyInputs = '';
  inputs.concat(output).forEach(x => {
    keyInputs += `${x.shape}_${x.getTextureShapeRC()}`;
  });
  const keyUserCode = program.userCode;
  const keyBroadcast = (program.supportsBroadcasting === true).toString();
  let key = program.constructor.name;
  // Fast string concat. See https://jsperf.com/string-concatenation/14.
  key += '_' + keyBroadcast + '_' + keyInputs + '_' + keyUserCode;
  return key;
}
