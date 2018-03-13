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
import { RegularArray } from "./types";

const debug = false;

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

export function objectsEqual(a: any, b: any): boolean {
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

const propelHosts = new Set(["", "127.0.0.1", "localhost", "propelml.org"]);

// Takes either a fully qualified url or a path to a file in the propel
// website directory. Examples
//
//    fetch("//tinyclouds.org/index.html");
//    fetch("deps/data/iris.csv");
//
// Propel files will use propelml.org if not being run in the project
// directory.
async function fetch2(p: string,
                      encoding: "binary" | "utf8" = "binary")
                      : Promise<string | ArrayBuffer> {
  // TODO The path hacks in this function are quite messy and need to be
  // cleaned up.
  p = fetch2ArgManipulation(p);
  if (IS_WEB) {
    const res = await fetch(p, { mode: "cors" });
    if (encoding === "binary") {
      return res.arrayBuffer();
    } else {
      return res.text();
    }
  } else {
    const { readFileSync } = nodeRequire("fs");
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

export function fetch2ArgManipulation(p: string): string {
  if (IS_WEB) {
    const host = document.location.host.split(":")[0];
    if (propelHosts.has(host)) {
      p = p.replace("deps/", "/");
      p = p.replace(/^src\//, "/src/");
    } else {
      p = p.replace("deps/", "http://propelml.org/");
      p = p.replace(/^src\//, "http://propelml.org/src/");
    }
  } else {
    const path = nodeRequire("path");
    if (!path.isAbsolute(p)) {
      p = path.join(__dirname, "..", p);
    }
  }
  return p;
}

export function randomString(): string {
  // 10 characters are returned:
  //   2log(36^10)                 ~= 51.7 bits
  //   mantisssa of IEEE754 double == 52 bits
  return (Math.random() + 1).toString(36).padEnd(12, "0").slice(2, 12);
}
