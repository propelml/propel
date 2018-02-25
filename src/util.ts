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
import { BasicTensor, DType, FlatVector, RegularArray, Shape,
  TensorLike, TypedArray } from "./types";

const debug = false;
const J = JSON.stringify;
const globalEval = eval;
export const IS_WEB = typeof window !== "undefined";
export const IS_NODE = !IS_WEB;
export const global = globalEval(IS_WEB ? "window" : "global");

if (IS_NODE) {
  // This is currently node.js specific.
  // TODO: move to api.js once it can be shared with the browser.
  const s = require("util").inspect.custom;
  Tensor.prototype[s] = function(depth: number, opts: {}) {
    return this.toString();
  };

  process.on("unhandledRejection", (error) => {
    throw error;
  });
}

export function allFinite(arr: number[]): boolean {
  for (const n of arr) {
    if (Number.isNaN(n)) return false;
  }
  return true;
}

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

export function log(...args: any[]) {
  if (debug) {
    console.log.apply(null, args);
  }
}

export function shapesEqual(x: Shape, y: Shape): boolean {
  if (x.length !== y.length) return false;
  for (let i = 0; i < x.length; ++i) {
    if (x[i] !== y[i]) return false;
  }
  return true;
}

export function assert(expr: boolean, msg = "") {
  if (!expr) {
    throw new Error(msg);
  }
}

export function assertFalse(expr: boolean, msg = "") {
  assert(!expr, msg);
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

// Provides a map with default value 0.
export class CounterMap {
  private map = new Map<number, number>();

  get(id: number): number {
    return this.map.has(id) ? this.map.get(id) : 0;
  }

  keys(): number[] {
    return Array.from(this.map.keys());

  }

  inc(id: number): void {
    this.map.set(id, this.get(id) + 1);
  }

  dec(id: number): void {
    this.map.set(id, this.get(id) - 1);
  }
}

export function deepCloneArray(arr: RegularArray<any>): typeof arr {
  const arr2 = [];
  for (let i = 0; i < arr.length; i++) {
    const value = arr[i];
    arr2[i] = Array.isArray(value) ? deepCloneArray(value) : value;
  }
  return arr2;
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

// Like setTimeout but with a promise.
export function delay(t: number): Promise<void> {
  return new Promise(function(resolve) {
    setTimeout(resolve, t);
  });
}

export function objectsEqual(a, b) {
  const aProps = Object.getOwnPropertyNames(a);
  const bProps = Object.getOwnPropertyNames(b);
  if (aProps.length !== bProps.length) return false;
  for (let i = 0; i < aProps.length; i++) {
    const k = aProps[i];
    if (a[k] !== b[k]) {
      return false;
    }
  }
  return true;
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

const propelHosts = new Set(["127.0.0.1", "localhost", "propelml.org"]);

// This is to confuse parcel.
// TODO There may be a more elegant workaround in future versions.
// https://github.com/parcel-bundler/parcel/pull/448
const nodeRequire = IS_WEB ? null : require;

// Takes either a fully qualified url or a path to a file in the propel
// website directory. Examples
//
//    fetch("//tinyclouds.org/index.html");
//    fetch("deps/data/iris.csv");
//
// Propel files will use propelml.org if not being run in the project
// directory.
async function fetch2(p: string,
    encoding: "binary" | "utf8" = "binary"): Promise<string | ArrayBuffer> {
  if (IS_WEB) {
    const host = document.location.host.split(":")[0];
    if (propelHosts.has(host)) {
      p = p.replace("deps/", "/");
    } else {
      p = p.replace("deps/", "http://propelml.org/");
    }
    const res = await fetch(p, { mode: "no-cors" });
    if (encoding === "binary") {
      return res.arrayBuffer();
    } else {
      return res.text();
    }
  } else {
    const { readFileSync } = nodeRequire("fs");
    const path = nodeRequire("path");
    if (!path.isAbsolute(p)) {
      p = path.join(__dirname, "..", p);
    }
    if (encoding === "binary") {
      const b = readFileSync(p, null);
      return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
    } else {
      return readFileSync(p, "utf8");
    }
  }
}

export async function fetchArrayBuffer(path: string): Promise<ArrayBuffer> {
  return fetch2(path, "binary") as any;
}

export async function fetchStr(path: string): Promise<string> {
  return fetch2(path, "utf8") as any;
}
