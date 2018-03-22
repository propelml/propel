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

// Node-only test. Browsers have caching built-in.

import * as fs from "fs";
import { test } from "../tools/tester";
import * as cache from "./cache";
import { assert, IS_NODE, nodeRequire } from "./util";
import { isDir } from "./util_node";

// Helper function to start a local web server.
// TODO should be moved to tools/tester eventually.
async function localServer(cb: (url: string) => Promise<void>): Promise<void> {
  if (!IS_NODE) {
    // We don't need a local server, since we're being hosted from one already.
    await cb(`http://${document.location.host}/`);
  } else {
    const root = __dirname + "/../build/dev_website";
    assert(isDir(root), root +
      " does not exist. Run ./tools/dev_website before running this test.");
    const { createServer } = nodeRequire("http-server");
    const server = createServer({ cors: true, root });
    server.listen();
    const port = server.server.address().port;
    const url = `http://127.0.0.1:${port}/`;
    try {
      await cb(url);
    } finally {
      server.close();
    }
  }
}

if (IS_NODE) {
  test(async function cache_url2Filename() {
    const actual = cache.url2Filename(
      "http://propelml.org/data/mnist/train-images-idx3-ubyte.bin");
    const expected0 =
      ".propel/cache/propelml.org/data/mnist/train-images-idx3-ubyte.bin";
    // Split and join done for windows compat.
    const { join } = nodeRequire("path");
    const expected = join(...expected0.split("/"));
    console.log("actual", actual);
    console.log("expected", expected);
    assert(actual.endsWith(expected));
  });
}

test(async function cache_fetchWithCache() {
  cache.clearAll();
  await localServer(async function(url: string) {
    url += "/data/mnist/train-images-idx3-ubyte.bin";
    const ab = await cache.fetchWithCache(url);
    assert(ab.byteLength === 47040016);
    if (IS_NODE) {
      assert(fs.existsSync(cache.url2Filename(url)));
    }
  });
});
