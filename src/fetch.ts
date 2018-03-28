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
  activeOutputHandler,
  Buffer,
  createResolvable,
  global,
  IS_NODE,
  IS_WEB,
  nodeRequire,
  process,
  randomString,
  URL,
} from "./util";

let lastProgress = 0;
const propelHosts = new Set(["", "127.0.0.1", "localhost", "propelml.org"]);

// Takes either a fully qualified url or a path to a file in the propel
// website directory. Examples
//
//    fetch("//tinyclouds.org/index.html");
//    fetch("deps/data/iris.csv");
//
// Propel files will use propelml.org if not being run in the project
// directory.
async function fetch2(p: string): Promise<ArrayBuffer> {
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
    if (global.PROPEL_TESTER) {
      p = p.replace("http://propelml.org/", "/");
    }
    try {
      const req = new XMLHttpRequest();
      const onLoad = createResolvable();
      req.onload = onLoad.resolve;
      req.onprogress = ev => downloadProgress(job, ev.loaded, ev.total);
      req.open("GET", p, true);
      req.responseType = "arraybuffer";
      req.send();
      await onLoad;
      return req.response;
    } finally {
      downloadProgress(job, null, null);
    }
  } else {
    if (p.match(/^http(s)*:\/\//)) {
      return fetchRemoteFile(p);
    }
    const path = nodeRequire("path");
    const { readFileSync } = nodeRequire("fs");
    if (!path.isAbsolute(p)) {
      p = path.join(__dirname, "..", p);
    }
    const b = readFileSync(p, null);
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  }
}

async function fetchRemoteFile(url: string): Promise<ArrayBuffer> {
  const u = new URL(url);

  // If we're in a testing environment, and trying to request
  // something from propelml.org, skip the download and get it from the repo.
  if (global.PROPEL_TESTER && u.hostname === "propelml.org") {
    const path = nodeRequire("path");
    url = path.join(__dirname, "../deps/", u.pathname);
    return fetch2(url);
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

  const b = Buffer.concat(chunks);
  return b.buffer.slice(b.byteOffset,
    b.byteOffset + b.byteLength) as ArrayBuffer;
}

export async function fetchArrayBuffer(path: string): Promise<ArrayBuffer> {
  return await fetch2(path);
}

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
