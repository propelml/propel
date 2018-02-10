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

/**
 * Returns the dimensions in the input shape that are broadcasted to
 * produce the provided output shape.
 *
 * The returned dimensions are 0-indexed and sorted. An example:
 * inShape = [4, 1, 3]
 * outShape = [5, 4, 3, 3]
 * result = [1]. Dimension 1 (2nd dimension of input) gets broadcasted 1 => 3.
 */
export function getBroadcastDims(
    inShape: number[], outShape: number[]): number[] {
  const inRank = inShape.length;
  const dims: number[] = [];
  for (let i = 0; i < inRank; i++) {
    const dim = inRank - 1 - i;
    const a = inShape[dim] || 1;
    const b = outShape[outShape.length - 1 - i] || 1;
    if (b > 1 && a === 1) {
      dims.unshift(dim);
    }
  }
  return dims;
}

/**
 * Given the output of `getBroadcastDims()`, returns true if the broadcasting
 * is along the outer-most dimensions of the input.
 */
export function broadcastDimsAreOuter(dims: number[]): boolean {
  for (let i = 0; i < dims.length; i++) {
    if (dims[i] !== i) {
      return false;
    }
  }
  return true;
}

export function assertAndGetBroadcastShape(
    shapeA: number[], shapeB: number[]): number[] {
  const result: number[] = [];
  const errMsg = `Operands could not be broadcast together with shapes ` +
      `${shapeA} and ${shapeB}.`;
  const l = Math.max(shapeA.length, shapeB.length);

  for (let i = 0; i < l; i++) {
    const a = shapeA[shapeA.length - i - 1] || 1;
    const b = shapeB[shapeB.length - i - 1] || 1;
    if (a > 1 && b > 1 && a !== b) {
      throw Error(errMsg);
    }
    result.unshift(Math.max(a, b));
  }
  return result;
}
