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

import { IS_NODE } from "./util";

const filterExpr = IS_NODE ? process.argv[2]
                           : new URL(location.href).hash.slice(1);
const filterRegExp = filterExpr ? new RegExp(filterExpr, "i") : null;
const tests = [];

function filter(fn) {
  if (filterRegExp) {
    return filterRegExp.test(fn.name);
  } else {
    return true;
  }
}

export function test(fn) {
  if (!fn.name) {
    throw new Error("Test function may not be anonymous");
  }
  if (filter(fn)) {
    tests.push(fn);
  }
}

async function runTests() {
  let passed = 0;
  let failed = 0;

  for (let i = 0; i < tests.length; i++) {
    const fn = tests[i];

    console.warn("%d/%d +%d -%d: %s",
                 i + 1,
                 tests.length,
                 passed,
                 failed,
                 fn.name);

    try {
      const r = await fn();
      passed++;
    } catch (e) {
      console.error((e && e.message) || e);
      failed++;
    }
  }

  console.warn(`DONE. passed: ${passed}, failed: ${failed}`);

  if (failed === 0) {
    // All good.
  } else if (IS_NODE) {
    process.exit(1);
  } else {
    // Use setTimeout to avoid the error being ignored due to unhandled
    // promise rejections being swallowed.
    setTimeout(() => {
      throw new Error(`There were ${failed} test failures.`);
    }, 0);
  }
}

setTimeout(runTests, 0);
