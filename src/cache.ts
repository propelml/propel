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

// When people load datasets in Propel, they need to be downloaded from HTTP.
// To avoid making people download the same dataset every time they start
// a training program, we provide a local cache of these datasets.
// The $HOME/.propel/cache directory is where these files will be stored.

import * as rimraf from "rimraf";
import { fetchArrayBuffer } from "./fetch";
import { assert, Buffer, IS_WEB, nodeRequire, URL } from "./util";
import { mkdirp, propelDir } from "./util_node";

export interface Cache {
  clearAll(): Promise<void>;
  get(url: string): Promise<undefined | ArrayBuffer>;
  set(url: string, ab: ArrayBuffer): Promise<void>;
}

let cacheImpl: Cache;

// TODO move this function to src/fetch.ts
export async function fetchWithCache(url: string): Promise<ArrayBuffer> {
  let ab = await cacheImpl.get(url);
  if (ab != null) {
    return ab;
  }
  ab = await fetchArrayBuffer(url);
  cacheImpl.set(url, ab);
  return ab;
}

export function clearAll(): Promise<void> {
  return cacheImpl.clearAll();
}

function cacheBase(): string {
  return nodeRequire("path").resolve(propelDir(), "cache");
}

// Maps a URL to a cache filename. Example:
// "http://propelml.org/data/mnist/train-images-idx3-ubyte.bin"
// "$HOME/.propel/cache/propelml.org/data/mnist/train-images-idx3-ubyte.bin"
export function url2Filename(url: string): string {
  // Throw on browser. We expose this method for testing, but only run it on
  // Node.
  assert(!IS_WEB, "url2Filename is unsupposed in the browser");
  const u = new URL(url);
  if (!(u.protocol === "http:" || u.protocol === "https:")) {
    throw Error(`Unsupported protocol '${u.protocol}'`);
  }
  // Just to be safe, use encodeURI to transform pathname and hostname. Note
  // that not all encoded components map to valid Windows filenames - there may
  // be a bug here in the future with URLs containing non-standard characters.
  const h = encodeURI(u.hostname);
  const p = encodeURI(u.pathname);
  assert(p.indexOf("..") < 0, "Safety sanity check");
  assert(h.indexOf("..") < 0, "Safety sanity check");
  const path = nodeRequire("path");
  // Note we purposely leave the port out of the cache path because
  // Windows doesn't allow colons in filenames. This is probably fine
  // in 99% of cases and is the simplest solution.
  const cacheFn = path.resolve(path.join(cacheBase(), h, p));
  assert(cacheFn.startsWith(cacheBase()));
  return cacheFn;
}

if (IS_WEB) {
  // On web do nothing. No caching.
  // Maybe use local storage?
  cacheImpl = {
    async clearAll(): Promise<void> { },
    async get(url: string): Promise<undefined | ArrayBuffer> {
      return undefined;
    },
    async set(url: string, ab: ArrayBuffer): Promise<void> { },
  };
} else {
  // Node caching uses the disk.
  const fs = nodeRequire("fs");
  const path = nodeRequire("path");

  cacheImpl = {
    async clearAll(): Promise<void> {
      rimraf.sync(cacheBase());
      console.log("Delete cache dir", cacheBase());
    },

    async get(url: string): Promise<undefined | ArrayBuffer> {
      const cacheFn = url2Filename(url);
      if (fs.existsSync(cacheFn)) {
        const b = fs.readFileSync(cacheFn, null);
        return b.buffer.slice(b.byteOffset,
            b.byteOffset + b.byteLength) as ArrayBuffer;
      } else {
        return undefined;
      }
    },

    async set(url: string, ab: ArrayBuffer): Promise<void> {
      const cacheFn = url2Filename(url);
      const cacheDir = path.dirname(cacheFn);
      mkdirp(cacheDir);
      fs.writeFileSync(cacheFn, Buffer.from(ab));
    },
  };
}
