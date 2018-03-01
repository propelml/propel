import { inspect } from "util";

export function generateMethods() {
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
}

type MatchTesters = {
  [name: string]: (value: any, ...args: any[]) => boolean;
};

export class PositiveMatchers /* implements Matchers */ {
  constructor(readonly actual: any) {}
  get not(): Matchers {
    const m = new NegativeMatchers(this.actual);
    return (m as any) as Matchers;
  }
}

class NegativeMatchers /* implements Matchers */ {
  constructor(readonly actual: any) {}
}

export const matchTesters: MatchTesters = {
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
      if (typeof a === "number" && typeof b === "number" &&
          isNaN(a) && isNaN(b)) {
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
