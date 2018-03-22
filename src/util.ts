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

export interface FetchEncodingMap {
  "arraybuffer": ArrayBuffer;
  "buffer": Buffer;
  "utf8": string;
}
export type FetchEncoding = keyof FetchEncodingMap;

// Takes either a fully qualified url or a path to a file in the propel
// website directory. Examples
//
//    fetch("//tinyclouds.org/index.html");
//    fetch("deps/data/iris.csv");
//
// Propel files will use propelml.org if not being run in the project
// directory.
async function fetch2<E extends FetchEncoding>(
    p: string, encoding: E): Promise<FetchEncodingMap[E]> {
  // TODO The path hacks in this function are quite messy and need to be
  // cleaned up.
  if (IS_WEB) {
    const job = randomString();
    const host = document.location.hostname;
    if (propelHosts.has(host)) {
      p = p.replace("deps/", "/");
      p = p.replace(/^src\//, "/src/");
    } else {
      p = p.replace("deps/", "http://propelml.org/");
      p = p.replace(/^src\//, "http://propelml.org/src/");
    }
    try {
      const req = new XMLHttpRequest();
      const onLoad = createResolvable();
      req.onload = onLoad.resolve;
      req.onprogress = ev => downloadProgress(job, ev.loaded, ev.total);
      req.open("GET", p, true);
      req.responseType = encoding === "utf8" ? "text" : "arraybuffer";
      if (encoding === "utf8") {
        req.overrideMimeType("text/plain; charset=utf-8");
      }
      req.send();
      await onLoad;
      return req.response;
    } finally {
      downloadProgress(job, null, null);
    }
  } else {
    if (p.match(/^http(s)*:\/\//)) {
      return fetchRemoteFile(p, encoding);
    }
    const path = nodeRequire("path");
    const { readFileSync } = nodeRequire("fs");
    if (!path.isAbsolute(p)) {
      p = path.join(__dirname, "..", p);
    }
    if (encoding === "buffer") {
      return readFileSync(p, null);
    } else if (encoding === "arraybuffer") {
      const b = readFileSync(p, null);
      return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
    } else {
      return readFileSync(p, "utf8");
    }
  }
}

async function fetchRemoteFile<E extends FetchEncoding>(
    url: string, encoding: E): Promise<FetchEncodingMap[E]> {
  const u = new URL(url);

  // If we're in a testing environment, and trying to request
  // something from propelml.org, skip the download and get it from the repo.
  if (global.PROPEL_TESTER && u.hostname === "propelml.org") {
    const path = nodeRequire("path");
    url = path.join(__dirname, "../deps/", u.pathname);
    return fetch2(url, encoding);
  }

  const http = nodeRequire(u.protocol === "https:" ? "https" : "http");
  const job = randomString();

  downloadProgress(job, 0, null); // Start download job with unknown size.

  const chunks: Buffer[] = [];

  try {
    const promise = createResolvable();
    const req = http.get(url, res => {
      const total = Number(res.headers["content-length"]);
      let loaded = 0;
      res.on("data", (chunk) => {
        chunks.push(chunk);
        loaded += chunk.length;
        downloadProgress(job, loaded, total);
      });
      res.on("end", promise.resolve);
      res.on("error", promise.reject);
    });
    req.on("error", promise.reject);
    await promise;

  } finally {
    downloadProgress(job, null, null); // End download job.
  }

  const buffer = Buffer.concat(chunks);
  if (encoding === "utf8") {
    return buffer.toString("utf8");
  } else {
    const b = buffer;
    return b.buffer.slice(b.byteOffset,
      b.byteOffset + b.byteLength) as ArrayBuffer;
  }
}

export async function fetchArrayBuffer(path: string): Promise<ArrayBuffer> {
  return await fetch2(path, "arraybuffer");
}

export async function fetchBuffer(path: string): Promise<Buffer> {
  if (IS_WEB) {
    throw new Error("`fetchBuffer` is not implemented to work on browser.");
  }
  return await fetch2(path, "buffer");
}

export async function fetchStr(path: string): Promise<string> {
  return await fetch2(path, "utf8");
}

export let activeOutputHandler: OutputHandler | null = null;

export function getOutputHandler(): OutputHandler | null {
  return activeOutputHandler;
}

export function setOutputHandler(handler: OutputHandler): void {
  activeOutputHandler = handler;
}

let lastProgress = 0;

export function downloadProgress(job: string, loaded: number | null,
                                 total: number | null): void {
  if (activeOutputHandler) {
    activeOutputHandler.downloadProgress({ job, loaded, total });
    return;
  }

  if (IS_NODE) {
    if (loaded === null && total === null) {
      // Write 7 spaces, so we can cover "100.00%".
      process.stdout.write(" ".repeat(7) + " \r");
    } else if (!total) {
      // Don't divide by zero.
      return;
    }

    const now = Date.now();
    if (now - lastProgress > 500) {
      // TODO: when multiple downloads are active, percentages currently
      // write over one another.
      const p = (loaded / total * 100).toFixed(2);
      process.stdout.write(`${p}% \r`);
      lastProgress = now;
    }
  }
}

export function randomString(): string {
  // 10 characters are returned:
  //   2log(36^10)                 ~= 51.7 bits
  //   mantisssa of IEEE754 double == 52 bits
  return (Math.random() + 1).toString(36).padEnd(12, "0").slice(2, 12);
}
