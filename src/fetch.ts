import { getCacheImpl } from "./cache";
import { createResolvable, downloadProgress,
         global, IS_WEB, nodeRequire, randomString, URL } from "./util";

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
    if (global.PROPEL_TESTER) {
      p = p.replace("http://propelml.org/", "/");
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

export async function fetchWithCache(url: string): Promise<ArrayBuffer> {
  const cacheImpl = getCacheImpl();
  let ab = await cacheImpl.get(url);
  if (ab != null) {
    return ab;
  }
  ab = await fetchArrayBuffer(url);
  cacheImpl.set(url, ab);
  return ab;
}
