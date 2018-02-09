declare function describe(name: string, fn: () => void): void;
declare function it(name: string, fn: (done: DoneFn) => void): void;
declare function beforeEach(action: HookFn): void;
declare function afterEach(action: HookFn): void;
declare function expect(actual: any): Matchers;
declare function fail(e?: any): void;
declare function spyOn(object: {}, methodName: string): Spy;

type HookFn = () => void;
type DoneFn = () => void;

interface Matchers {
  toBe(expected: any): void;
  toEqual(expected: any): void;
  toBeUndefined(): void;
  toBeNull(): void;
  toBeLessThan(expected: number): void;
  toBeLessThanOrEqual(expected: number): void;
  toBeGreaterThan(expected: number): void;
  toBeGreaterThanOrEqual(expected: number): void;
  toThrow(): void;
  toThrowError(message?: string | RegExp): void;
  not: Matchers;
}

interface Spy {
  and: SpyAnd;
}

interface SpyAnd {
  returnValue(val: any): void;
}
