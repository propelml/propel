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
const globalEval = eval;
export const IS_WEB = typeof window !== "undefined";
export const IS_NODE = !IS_WEB;
export const global = globalEval(IS_WEB ? "window" : "global");

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

const propelHosts = new Set(["127.0.0.1", "localhost", "propelml.org"]);

// This is to confuse parcel.
// TODO There may be a more elegant workaround in future versions.
// https://github.com/parcel-bundler/parcel/pull/448
export const nodeRequire = IS_WEB ? null : require;

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
  // TODO The path hacks in this function are quite messy and need to be
  // cleaned up.
  p = fetch2ArgManipulation(p);
  if (IS_WEB) {
    const res = await fetch(p, { mode: "no-cors" });
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
