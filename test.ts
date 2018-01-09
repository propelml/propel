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

// tslint:disable-next-line:no-reference
/// <reference path='./test.d.ts' />

import { inspect } from "util";
import { assert, global, IS_NODE } from "./util";

const filterExpr = IS_NODE ? process.argv[2]
                           : new URL(location.href).hash.slice(1);
const filterRegExp = filterExpr ? new RegExp(filterExpr, "i") : null;
const tests = [];

function filter(name) {
  if (filterRegExp) {
    return filterRegExp.test(name);
  } else {
    return true;
  }
}

export function test(fn) {
  const name = fn.name;
  if (!name) {
    throw new Error("Test function may not be anonymous");
  }
  if (filter(name)) {
    tests.push({ fn, name });
  }
}

const matchers = {
  toBe(value, expected) {
    return value === expected;
  },
  toBeNull(value) {
    return value === null;
  },
  toBeUndefined(value) {
    return value === undefined;
  },
  toEqual(value, expected) {
    const seen = new Map();
    return (function compare(a, b) {
      if (a === b) {
        return true;
      }
      if (isNaN(a) && isNaN(b)) {
        return true;
      }
      if (a && typeof a === "object" &&
          b && typeof b === "object") {
        if (seen.get(a) === b) {
          return true;
        }
        for (const key in { ...a, ...b }) {
          if (!compare(a[key], b[key])) {
            return false;
          }
        }
        seen.set(a, b);
        return true;
      }
      return false;
    })(value, expected);
  },
  toBeLessThan(value, comparand) {
    return value < comparand;
  },
  toBeLessThanOrEqual(value, comparand) {
    return value <= comparand;
  },
  toBeGreaterThan(value, comparand) {
    return value > comparand;
  },
  toBeGreaterThanOrEqual(value, comparand) {
    return value >= comparand;
  },
  toThrow(fn) {
    try {
      fn();
      return false;
    } catch (e) {
      return true;
    }
  },
  toThrowError(fn, message) {
    let error;
    try {
      fn();
    } catch (e) {
      error = e;
    }
    return error instanceof Error &&
           (message === undefined || message.match(error.message));
  }
};

export class Expector {
  value: any;
  constructor(value) {
    this.value = value;
  }
  get not() {
    return new NotExpector(this.value);
  }
}
export class NotExpector extends Expector {
}

for (const name in matchers) {
  if (!matchers.hasOwnProperty(name)) {
    continue;
  }

  const fn = matchers[name];
  Expector.prototype[name] = function(...args) {
    if (!fn(this.value, ...args)) {
      throw new Error(`Expected ` + inspect(this.value) + " " + name +
                      [...args].map((v) => " " + inspect(v)).join(""));
    }
  };
  NotExpector.prototype[name] = function(...args) {
    if (fn(this.value, ...args)) {
      throw new Error(`Didn't expect ` + inspect(this.value) + " " + name +
                      [...args].map((v) => " " + inspect(v)).join(""));
    }
  };
}

export const expect = global.expect = function expect(value) {
  return new Expector(value);
};

export const describe = global.describe = (groupName, fn) => {
  let beforeEachFn = () => {};
  let afterEachFn = () => {};

  global.beforeEach = (fn) => beforeEachFn = fn;
  global.afterEach = (fn) => afterEachFn = fn;

  global.it = (name, fn) => {
    name = `${groupName}: ${name}`;
    if (!filter(name)) {
      return;
    }

    let wrapper;
    if (fn.length) {
      // `fn` is expecting a `done` callback;
      wrapper = async() => {
        let done;
        const promise = new Promise((res, rej) => { done = res; });
        beforeEachFn();
        try {
          fn(done);
          await promise;
        } finally {
          afterEachFn();
        }
      };
    } else {
      wrapper = () => {
        beforeEachFn();
        try {
          fn();
        } finally {
          afterEachFn();
        }
      };
    }
    tests.push({ fn: wrapper, name });
  };

  fn();

  global.it = null;
};

async function runTests() {
  let passed = 0;
  let failed = 0;

  for (let i = 0; i < tests.length; i++) {
    const { fn, name } = tests[i];

    try {
      const r = await fn();
      passed++;
    } catch (e) {
      failed++;
      console.warn("%d/%d +%d -%d: %s",
                 i + 1,
                 tests.length,
                 passed,
                 failed,
                 name);
      console.error(e && e.message || e);
    }
  }

  console.warn(`DONE. passed: ${passed}, failed: ${failed}`);

  if (failed === 0) {
    // All good.
  } else if (IS_NODE) {
    process.exit(1);
  } else {
    throw new Error(`There were ${failed} test failures.`);
  }
}

setTimeout(runTests, 0);
