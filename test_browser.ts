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
import { createServer } from "http-server";
import * as puppeteer from "puppeteer";
import { format } from "util";
import "./util"; // To make unhandled rejections crash node.

// The PP_TEST_DEBUG environment variable can be used to run the tests in
// debug mode. When debug mode is enabled...
//   - the browser runs in interactive (not headless) mode.
//   - tabs stay open after running the tests.
//   - the built-in webserver remains online after running the tests.
// Hit 'enter' in the terminal to stop the webserver and exit the browser.
const debug = !!process.env.PP_TEST_DEBUG;

// When the PP_TEST_DL environment variable is set, the DeepLearn test suite is
// also executed. Since webgl isn't available in headless mode, this flag
// causes Chromium to run in interactive mode.
const testdl = !!process.env.PP_TEST_DL;

const TESTS = [
  // This page loads and runs all the webpack'ed unit tests.
  // The test harness logs "DONE bla bla" to the console when done.
  // If this message doesn't appear, or an unhandled error is thrown on the
  // page, the test fails.
  {
    doneMsg: /^DONE.*failed: 0/,
    href: "static/test.html#script=/test_isomorphic.js",
    timeout: 40 * 1000
  },

  // These web pages are simply loaded; the test passes if no unhandled
  // exceptions are thrown on the page.
  { href: "index.html" },
  { href: "notebook/" },
  { href: "notebook/?nbId=default" },
  { href: "docs/index.html" },
];

if (testdl) {
  TESTS.unshift({
    doneMsg: /^DONE.*failed: 0/,
    href: "static/test.html#script=/test_dl.js",
    timeout: 2 * 60 * 1000
  });
}

(async() => {
  let passed = 0, failed = 0;

  const server = createServer({ root: "./build/website" });
  server.listen();
  const port = server.server.address().port;

  const browser = await puppeteer.launch({
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
    headless: !debug && !testdl
  });

  for (let i = 0; i < TESTS.length; i++) {
    const test = TESTS[i];
    const url = `http://localhost:${port}/${test.href}`;
    if (await runTest(browser, url, test as any)) {
      passed++;
    } else {
      failed++;
    }
  }

  if (debug) {
    await new Promise((res) => process.stdin.once("data", res));
  }

  await browser.close();
  server.close();

  console.log(`DONE. passed: ${passed}, failed: ${failed}`);
  if (failed > 0) {
    process.exit(1);
  }
})();

function prefix(s: string, prefix: string): string {
  return prefix + s.replace(/\n/g, "\n" + prefix);
}

// `doneMsg` may be a string or a regex. If it is specified, the web page
// must have a matching message logged to the console, otherwise the test
// is considered to have failed. If doneMsg is set to null, the test
// will be considered to have passed if no errors are thrown before the
// time-out expires.
async function runTest(browser, url, { href, doneMsg = null, timeout = 1000 }) {
  let pass, fail;
  const promise = new Promise((res, rej) => { pass = res; fail = rej; });
  let timer = null;
  const timeoutIsFailure = doneMsg != null;

  console.log(`TEST: ${href}`);

  const page = await browser.newPage();
  page.on("load", onLoad);
  page.on("console", onMessage);
  page.on("response", onResponse);
  page.on("pageerror", onError);
  page.goto(url);

  try {
    await promise;
    console.log(`PASS: ${href}\n`);
    return true;
  } catch (err) {
    console.log(err.message); // Stack trace is useless here.
    console.log(`FAIL: ${href}\n`);
    return false;
  } finally {
    if (!debug) await page.close();
    cancelTimer();
  }

  function onLoad() {
    restartTimer();
  }

  function onError(browserError) {
    const err = new Error(prefix(browserError.message, "> "));
    fail(err);
  }

  function onResponse(res) {
    // Puppeteer sets res.ok if the HTTP status is in the 2xx range.
    // Additionally we want to treat status 304 ("Not Modified") as a success.
    if (!(res.ok || res.status === 304)) {
      fail(new Error(`HTTP ${res.status}: ${res.url}`));
    }
  }

  function onTimeOut() {
    if (timeoutIsFailure) {
      fail(new Error(`Timeout (${timeout}ms)`));
    } else {
      pass();
    }
  }

  function onMessage(msg) {
    const values = msg.args.map(
      v =>
        v._remoteObject.value !== undefined
          ? v._remoteObject.value
          : `[[${v._remoteObject.type}]]`
    );
    const text = format.apply(null, values);

    console.log(prefix(text, "> "));

    if (doneMsg != null && text.match(doneMsg)) {
      pass();
    } else {
      restartTimer();
    }
  }

  function restartTimer() {
    cancelTimer();
    timer = setTimeout(onTimeOut, timeout);
  }

  function cancelTimer() {
    if (timer != null) {
      clearTimeout(timer);
      timer = null;
    }
  }
}
