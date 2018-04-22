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
import { OutputHandler } from "./output_handler";
import { RegularArray } from "./types";

export const debug = false;

// If you use the eval function indirectly, by invoking it via a reference
// other than eval, as of ECMAScript 5 it works in the global scope rather than
// the local scope. This means, for instance, that function declarations create
// global functions, and that the code being evaluated doesn't have access to
// local variables within the scope where it's being called.
export const globalEval = eval;

// A reference to the global object.
export const global = globalEval("this");

export const IS_WEB = global.window !== undefined;
export const IS_NODE = !IS_WEB;

// Parcel tries to be smart and replaces the process and Buffer object.
export const process = IS_NODE ? globalEval("process") : null;
// tslint:disable-next-line:variable-name
export const Buffer = IS_NODE ? globalEval("Buffer") : null;

if (IS_NODE) {
  process.on("unhandledRejection", (error) => {
    throw error;
  });
}

// This is to confuse parcel and prevent it from including node-only modules
// in a browser-targeted bundle.
// TODO: There may be a more elegant workaround in future versions.
// https://github.com/parcel-bundler/parcel/pull/448
export const nodeRequire = IS_WEB ? null : require;

// WHATWG-standard URL class.
export const URL = IS_WEB ? window.URL : nodeRequire("url").URL;

export function log(...args: any[]) {
  if (debug) {
    console.log.apply(null, args);
  }
}

export function assert(expr: boolean, msg = "") {
  if (!expr) {
    throw new Error(msg);
  }
}

export function equal(c: any, d: any): boolean {
  const seen = new Map();
  return (function compare(a, b) {
    if (a === b) {
      return true;
    }
    if (typeof a === "number" && typeof b === "number" &&
        isNaN(a) && isNaN(b)) {
      return true;
    }
    if (a && typeof a === "object" && b && typeof b === "object") {
      if (seen.get(a) === b) {
        return true;
      }
      if (Object.keys(a).length !== Object.keys(b).length) {
        return false;
      }
      for (const key in { ...a, ...b }) {
        if (!compare(a[key], b[key])) {
          return false;
        }
      }
      seen.set(a, b);
      return true;
    }
    return false;
  })(c, d);
}

export function assertEqual(actual: any, expected: any, msg = null) {
  if (!msg) { msg = `actual: ${actual} expected: ${expected}`; }
  if (!equal(actual, expected)) {
    console.error(
      "assertEqual failed. actual = ", actual, "expected =", expected);
    throw new Error(msg);
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

// Like setTimeout but with a promise.
export function delay(t: number): Promise<void> {
  return new Promise(function(resolve) {
    setTimeout(resolve, t);
  });
}

// A `Resolvable` is a Promise with the `reject` and `resolve` functions
// placed as methods on the promise object itself. It allows you to do:
//
//   const p = createResolvable<number>();
//   ...
//   p.resolve(42);
//
// It'd be prettier to make Resolvable a class that inherits from Promise,
// rather than an interface. This is possible in ES2016, however typescript
// produces broken code when targeting ES5 code.
// See https://github.com/Microsoft/TypeScript/issues/15202
// At the time of writing, the github issue is closed but the problem remains.
export interface Resolvable<T> extends Promise<T> {
  resolve: (value?: T | PromiseLike<T>) => void;
  reject: (reason?: any) => void;
}
export function createResolvable<T>(): Resolvable<T> {
  let methods;
  const promise = new Promise<T>((resolve, reject) => {
    methods = { resolve, reject };
  });
  return Object.assign(promise, methods) as Resolvable<T>;
}

let activeOutputHandler: OutputHandler | null = null;

export function getOutputHandler(): OutputHandler | null {
  return activeOutputHandler;
}

export function setOutputHandler(handler: OutputHandler): void {
  activeOutputHandler = handler;
}

export function randomString(): string {
  // 10 characters are returned:
  //   2log(36^10)                 ~= 51.7 bits
  //   mantisssa of IEEE754 double == 52 bits
  return (Math.random() + 1).toString(36).padEnd(12, "0").slice(2, 12);
}

// The spec says that the maximum length of an array is UINT32_MAX, or 2**32-1.
// Since the array length always equals one plus the key of the last element,
// the maximum valid key is UINT32_MAX-1 or 2**32-2.
const maxNumericalKey = 2 ** 32 - 2;

// Array, string, TypedArray indexes are actually strings, and not numbers.
// However properties keyed by these "numerical strings" do behave differently.
// This function returns true when a string (key looks like be a numerical key.
export function isNumericalKey(key: string): boolean {
  return Number(key) <= maxNumericalKey &&
         (key === "0" || /^[1-9]\d*$/.test(key));
}

export function tmpdir(): string {
  return process.env.TEMP || process.env.TMPDIR || "/tmp";
}

// TODO maybe share this with nb_transpiler.getEntryPoint().
export function captureStackTrace(error?: Error): string {
  let stack: string;
  if (error) {
    stack = error.stack;
  } else {
    try {
      throw new Error();
    } catch (e) {
      stack = e.stack;
    }
  }
  return stack;
}

export function formatImageName(filename: string, i?: number) {
  if (typeof filename !== "string") {
    const random = randomString().substr(0, 6);
    filename = `propel-${random}`;
  }
  // Force a valid file extension at the end of `filename` (default to .png)
  const validExtensions = [".png", ".jpg", ".jpeg"];
  const tmp = filename.toLowerCase();
  let insert = true;
  for (const ext of validExtensions) {
    if (tmp.endsWith(ext)) {
      insert = false;
      break;
    }
  }
  if (insert) filename += ".png";

  if (typeof i === "number") {
    const index = filename.lastIndexOf(".");
    return filename.slice(0, index) + `-${i}` + filename.slice(index);
  }

  return filename;
}
