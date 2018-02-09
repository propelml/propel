// tslint:disable-next-line:no-reference
/// <reference path='./jasmine_types.d.ts' />

import { inspect } from "util";
import { global } from "../src/util";
import { test } from "./tester";

class Suite {
  constructor(readonly name: string) {}
  before: HookFn[] = [];
  after: HookFn[] = [];
}

let currentSuite: Suite = null;

interface SpyInfo {
  object: any;
  methodName: string;
  method: Function;
}

let spies: SpyInfo[] = [];

type MatchTesters = {
  [name: string]: (value: any, ...args: any[]) => boolean;
};

const matchTesters: MatchTesters = {
  toBe(value: any, expected: any): boolean {
    return value === expected;
  },

  toBeNull(value: any): boolean {
    return value === null;
  },

  toBeUndefined(value: any): boolean {
    return value === undefined;
  },

  toEqual(value: any, expected: any): boolean {
    const seen = new Map();
    return (function compare(a, b) {
      if (a === b) {
        return true;
      }
      if (isNaN(a) && isNaN(b)) {
        return true;
      }
      if (a && typeof a === "object" && b && typeof b === "object") {
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

  toBeLessThan(value: number, comparand: number): boolean {
    return value < comparand;
  },

  toBeLessThanOrEqual(value: number, comparand: number): boolean {
    return value <= comparand;
  },

  toBeGreaterThan(value: number, comparand: number): boolean {
    return value > comparand;
  },

  toBeGreaterThanOrEqual(value: number, comparand: number): boolean {
    return value >= comparand;
  },

  toThrow(fn: () => void): boolean {
    try {
      fn();
      return false;
    } catch (e) {
      return true;
    }
  },

  toThrowError(fn: () => void, message: string | RegExp): boolean {
    let error;
    try {
      fn();
    } catch (e) {
      error = e;
    }
    return (
      error instanceof Error &&
      (message === undefined || error.message.match(message) !== null)
    );
  }
};

class PositiveMatchers /* implements Matchers */ {
  constructor(readonly actual: any) {}
  get not(): Matchers {
    const m = new NegativeMatchers(this.actual);
    return (m as any) as Matchers;
  }
}

class NegativeMatchers /* implements Matchers */ {
  constructor(readonly actual: any) {}
}

for (const name of Object.keys(matchTesters)) {
  const pos = PositiveMatchers.prototype as any;
  const neg = NegativeMatchers.prototype as any;
  const testFn = matchTesters[name];
  pos[name] = function(...args: any[]) {
    if (!testFn(this.actual, ...args)) {
      const msg =
        `Expected ${inspect(this.actual)} ` +
        [name, ...args.map(a => inspect(a))].join(" ");
      throw new Error(msg);
    }
  };
  neg[name] = function(...args: any[]) {
    if (testFn(this.actual, ...args)) {
      const msg =
        `Didn't expect ${inspect(this.actual)} ` +
        [name, ...args.map(a => inspect(a))].join(" ");
      throw new Error(msg);
    }
  };
}

global.expect = function expect(actual: any): Matchers {
  const m = new PositiveMatchers(actual);
  return (m as any) as Matchers;
};

global.fail = function(e?: any): void {
  if (!(e instanceof Error)) e = new Error(e);
  throw e;
};

global.describe = function describe(name: string, fn: () => void): void {
  currentSuite = new Suite(name);
  fn();
  currentSuite = null;
};

global.it = function it(name: string, fn: (done?: DoneFn) => void): void {
  const suite = currentSuite;
  const isAsync = fn.length > 0;
  name = `${suite.name}: ${name}`;

  const wrapper = async() => {
    callHooks(suite.before);
    try {
      if (isAsync) {
        let done;
        const promise = new Promise((res, rej) => {
          done = res;
        });
        fn(done);
        await promise;
      } else {
        fn();
      }
    } finally {
      callHooks(suite.after);
      removeSpies();
    }
  };

  test({ fn: wrapper, name });
};

global.beforeEach = function beforeEach(fn: HookFn) {
  currentSuite.before.push(fn);
};

global.afterEach = function afterEach(fn: HookFn) {
  currentSuite.after.unshift(fn);
};

global.spyOn = function spyOn(object: {}, methodName: string): Spy {
  return {
    and: {
      returnValue(retVal) {
        const o = object as any;
        spies.push({ object: o, methodName, method: o[methodName] });
        o[methodName] = () => retVal;
      }
    }
  };
};

function callHooks(hooks: HookFn[]) {
  for (const fn of hooks) {
    fn();
  }
}

function removeSpies() {
  for (const { object, methodName, method } of spies) {
    object[methodName] = method;
  }
  spies = [];
}
