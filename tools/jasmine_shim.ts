// tslint:disable-next-line:no-reference
/// <reference path='./jasmine_types.d.ts' />

import { global } from "../src/util";
import { PositiveMatchers, runTests } from "./jasmine_shim_testers";
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

generateMethods();

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
