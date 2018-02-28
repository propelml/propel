/*!
   Copyright 2018 Propel http://propel.site/.  All rights reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

import { Tensor } from "./tensor";
import { BasicTensor, DType, FlatVector, RegularArray, Shape, TensorLike,
    TypedArray } from "./types";
import { assert } from "./util";
export { assert } from "./util";

function toShapeAndFlatVector(t: TensorLike): [Shape, FlatVector] {
  if ((t as Tensor).cpu) {
    t = (t as Tensor).cpu(); // Copy to CPU if necessary.
  }
  if ((t as BasicTensor).dataSync) {
    t = t as BasicTensor;
    return [t.shape, t.dataSync()];
  } else if (isTypedArray(t)) {
    return [[t.length], t];
  } else if (t instanceof Array) {
    return [inferShape(t), flatten(t) as number[]];
  } else if (typeof t === "number") {
    return [[], [t]];
  } else {
    throw new Error("Not TensorLike");
  }
}

function toNumber(t: TensorLike): number {
  const values = toShapeAndFlatVector(t)[1];
  if (values.length !== 1) {
    throw new Error("Not Scalar");
  }
  return values[0];
}

export function shapesEqual(x: Shape, y: Shape): boolean {
  if (x.length !== y.length) return false;
  for (let i = 0; i < x.length; ++i) {
    if (x[i] !== y[i]) return false;
  }
  return true;
}

export function allFinite(arr: number[]): boolean {
  for (const n of arr) {
    if (Number.isNaN(n)) return false;
  }
  return true;
}

export function assertClose(actual: TensorLike, expected: TensorLike,
                            delta = 0.001) {
  actual = toNumber(actual);
  expected = toNumber(expected);
  assert(Math.abs(actual - expected) < delta,
    `actual: ${actual} expected: ${expected}`);
}

export function assertEqual(actual: TensorLike, expected: number | boolean,
                            msg = null) {
  actual = toNumber(actual);
  if (!msg) { msg = `actual: ${actual} expected: ${expected}`; }
  assert(actual === expected, msg);
}

export function assertShapesEqual(actual: Shape, expected: Shape) {
  const J = JSON.stringify;
  const msg = `Shape mismatch. actual: ${J(actual)} expected ${J(expected)}`;
  assert(shapesEqual(actual, expected), msg);
}

export function assertAllEqual(actual: TensorLike, expected: TensorLike) {
  const [actualShape, actualFlat] = toShapeAndFlatVector(actual);
  const [expectedShape, expectedFlat] = toShapeAndFlatVector(expected);
  assertShapesEqual(actualShape, expectedShape);
  for (let i = 0; i < actualFlat.length; i++) {
    assert(actualFlat[i] === expectedFlat[i],
      `index ${i} actual: ${actualFlat[i]} expected: ${expectedFlat[i]}`);
  }
}

export function assertAllClose(actual: TensorLike, expected: TensorLike,
                               delta = 0.001) {
  const [actualShape, actualFlat] = toShapeAndFlatVector(actual);
  const [expectedShape, expectedFlat] = toShapeAndFlatVector(expected);

  assertShapesEqual(actualShape, expectedShape);

  for (let i = 0; i < actualFlat.length; ++i) {
    const a = (actualFlat[i]) as number;
    const e = (expectedFlat[i]) as number;
    assert(Math.abs(a - e) < delta,
      `index ${i} actual: ${actualFlat[i]} expected: ${expectedFlat[i]}`);
  }
}

export function bcastGradientArgs(sx: Shape, sy: Shape): [Shape, Shape] {
  const rx = [];
  const ry = [];
  const m = Math.max(sx.length, sy.length);
  for (let i = 0; i < m; ++i) {
    const argIndex = m - i - 1;
    if (i >= sx.length) {
      rx.unshift(argIndex);
      continue;
    }
    if (i >= sy.length) {
      ry.unshift(argIndex);
      continue;
    }

    const xDim = sx[sx.length - i - 1];
    const yDim = sy[sy.length - i - 1];

    if (xDim === yDim) continue;

    if (xDim === 1 && yDim !== 1) {
      rx.unshift(argIndex);
    } else if (xDim !== 1 && yDim === 1) {
      ry.unshift(argIndex);
    } else {
      assert(xDim === 1 && yDim === 1, "Incompatible broadcast shapes.");
    }
  }
  return [rx, ry];
}

export function isTypedArray(x: any): x is TypedArray {
  return (x instanceof Float32Array || x instanceof Uint8Array ||
          x instanceof Int32Array);
}

export function getDType(data: TypedArray): DType {
  if (data instanceof Int32Array) {
    return "int32";
  } else if (data instanceof Float32Array) {
    return "float32";
  } else if (data instanceof Uint8Array) {
    return "uint8";
  } else {
    throw new Error("Unsupported TypedArray flavor");
  }
}

export function makeTypedArray(data, dtype: DType = "float32"): TypedArray {
  switch (dtype) {
    case "bool":
      return new Uint8Array(data);
    case "float32":
      return new Float32Array(data);
    case "int32":
      return new Int32Array(data);
    case "uint8":
      return new Uint8Array(data);
    default:
      throw new Error("Not implemented");
  }
}

/**
 * flatten() and inferShape were forked from deps/deeplearnjs/src/util.ts
 *
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

export function flatten(
    arr: number | boolean | RegularArray<number> | RegularArray<boolean>,
    ret: Array<number | boolean> = []): Array<number | boolean> {
  if (Array.isArray(arr)) {
    for (let i = 0; i < arr.length; ++i) {
      flatten(arr[i], ret);
    }
  } else {
    ret.push(arr);
  }
  return ret;
}

export function inferShape(arr: number | boolean | RegularArray<number> |
                           RegularArray<boolean>): number[] {
  const shape: number[] = [];
  while (arr instanceof Array) {
    shape.push(arr.length);
    arr = arr[0];
  }
  return shape;
}
