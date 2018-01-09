declare function describe(
  description: string,
  specDefinitions: () => void
): void;
declare function it(
  expectation: string,
  assertion?: (done: DoneFn) => void
): void;

declare function beforeEach(action: () => void): void;
declare function afterEach(action: () => void): void;

declare function expect<T>(actual: T): Matchers<T>;
declare function fail(e?: any): void;

type DoneFn = () => void;

type Expected<T> = any;

interface Matchers<T> {
  toBe(expected: Expected<T>): boolean;
  toEqual(expected: Expected<T>): boolean;
  toBeUndefined(): boolean;
  toBeNull(): boolean;
  toBeLessThan(expected: number): boolean;
  toBeLessThanOrEqual(expected: number): boolean;
  toBeGreaterThan(expected: number): boolean;
  toBeGreaterThanOrEqual(expected: number): boolean;
  toThrow(): boolean;
  toThrowError(message?: string | RegExp): boolean;

  not: Matchers<T>;
}
