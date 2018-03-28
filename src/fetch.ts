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

// This file contains Propel's central system for loading data from remote URLs
// or local files. There is a progress system to notify users about download.
// See also fetchWithCache.
import {
  assert,
  Buffer,
  createResolvable,
  getOutputHandler,
  global,
  IS_NODE,
  IS_WEB,
  nodeRequire,
  process,
  randomString,
  URL,
} from "./util";

export const propelHostname = "propelml.org";
export const propelURL = `http://${propelHostname}/`;
let lastProgress = 0;

export function fetchArrayBuffer(p: string): Promise<ArrayBuffer> {
  const url = resolve(p);
  const job = randomString();
  downloadProgress(job, 0, null); // Start download job with unknown size.
  try {
    if (IS_WEB) {
      return fetchBrowserXHR(job, url);
    } else if (isHTTP(url)) {
      return fetchNodeHTTP(job, url);
    } else {
      return fetchNodeFS(job, url);
    }
  } finally {
    downloadProgress(job, null, null);
  }
}

// Transforms URLs that point to local resources to paths
// matching those resources.
export function resolve(p: string, isTest?: boolean): URL {
  // If the optional isTest argument isn't supplied look for the global
  // variable PROPEL_TESTER.
  if (isTest == null) isTest = !!global.PROPEL_TESTER;

  let base: URL;
  if (IS_WEB) {
    // Hack for tools/website_render.js
    if (document.location.protocol === "about:") {
      base = new URL("http://localhost:8080/");
    } else {
      base = new URL(document.location);
    }
  } else {
    base = new URL("file:///");
    base.pathname = process.cwd() + "/";
  }

  let url = new URL(p, base);

  // If we're in a testing environment, and trying to request
  // something from propelml.org, skip the download and get it from the repo.
  if (isTest && url.hostname === propelHostname) {
    if (IS_WEB) {
      url.host = document.location.host;
    } else {
      const url2 = new URL("file:///");
      const { join } = nodeRequire("path");
      const pathname = join(__dirname, "../build/dev_website/", url.pathname);
      url2.pathname = pathname;
      url = url2;
    }
  }
  return url;
}

async function fetchBrowserXHR(job: string, url: URL): Promise<ArrayBuffer> {
  assert(isHTTP(url));
  const req = new XMLHttpRequest();
  const promise = createResolvable();
  req.onload = promise.resolve;
  req.onprogress = ev => downloadProgress(job, ev.loaded, ev.total);
  req.open("GET", url.toString(), true);
  req.responseType = "arraybuffer";
  req.send();
  await promise;
  return req.response;
}

async function fetchNodeHTTP(job: string, url: URL): Promise<ArrayBuffer> {
  assert(isHTTP(url));
  const http = nodeRequire(url.protocol === "https:" ? "https" : "http");
  const chunks: Buffer[] = [];
  const promise = createResolvable();
  const req = http.get(url.toString(), res => {
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
  const b = Buffer.concat(chunks);
  return b.buffer.slice(b.byteOffset,
    b.byteOffset + b.byteLength) as ArrayBuffer;
}

async function fetchNodeFS(job: string, url: URL): Promise<ArrayBuffer> {
  // This function is async for consistancy with fetchNodeHTTP and
  // fetchBrowserXHR, even tho it is actually sync.
  const fs = nodeRequire("fs");
  let p = decodeURIComponent(url.pathname);
  if (process.platform === "win32" && p.startsWith("/")) {
    p = p.slice(1);
  }
  const b = fs.readFileSync(p, null);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

export function downloadProgress(job: string, loaded: number | null,
                                 total: number | null): void {
  const oh = getOutputHandler();
  if (oh != null) {
    oh.downloadProgress({ job, loaded, total });
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

function isHTTP(url: URL): boolean {
  return url.protocol === "http:" || url.protocol === "https:";
}
