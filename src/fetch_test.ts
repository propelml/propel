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

import {
  localServer,
  test,
} from "../tools/tester";
import * as fetch from "./fetch";
import {
  assert,
  IS_WEB,
  nodeRequire,
  process,
  URL,
} from "./util";

const mnistPath = "/data/mnist/t10k-images-idx3-ubyte.bin";

test(async function fetch_fetchArrayBuffer() {
  await localServer(async function(url: string) {
    url += mnistPath;
    const ab = await fetch.fetchArrayBuffer(url);
    assert(ab.byteLength === 7840016);
  });
});

test(async function fetch_relativeWithSpacesNode() {
  if (IS_WEB) return;
  const origDir = process.cwd();
  try {
    // chdir to propel root, so we can test relative access.
    process.chdir(__dirname + "/..");
    const ab = await fetch.fetchArrayBuffer(
      "src/testdata/dir with spaces/hello.txt");
    assert(ab.byteLength === 7);
  } finally {
    process.chdir(origDir);
  }
});

test(function fetch_resolve() {
  // When in testing mode, MNIST should resolve to local paths.
  const p = "http://propelml.org" + mnistPath;
  let actual = fetch.resolve(p, true).toString();
  let expected: string;
  if (IS_WEB) {
    expected = document.location.origin +
      "/data/mnist/t10k-images-idx3-ubyte.bin";
  } else {
    const expectedURL = new URL("file:///");
    expectedURL.pathname = nodeRequire("path").resolve(__dirname,
      "../build/dev_website" + mnistPath);
    expected = expectedURL.toString();
  }
  assert(actual === expected);
  // When not in testing mode, the URL shouldn't be modified.
  actual = fetch.resolve(p, false).toString();
  expected = p;
  assert(actual === expected);
});

test(function fetch_resolveRelativeSpaces() {
  const p = "relative/path/with spaces/file.npy";
  const actual = fetch.resolve(p).toString();
  let expected: string;
  if (IS_WEB) {
    const base = document.location.toString();
    expected = (new URL(p, base)).toString();
  } else {
    const expectedURL = new URL("file:///");
    expectedURL.pathname = nodeRequire("path").resolve(p);
    expected = expectedURL.toString();
  }
  console.log("actual", actual);
  console.log("expected", expected);
  assert(actual === expected);
});
