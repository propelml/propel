import { TensorLike, BasicTensor, isTypedArray, FlatVector, Shape }
  from "./types";
import { flatten, inferShape } from "./deeplearnjs/src/util";

const debug = false;
const J = JSON.stringify;

function toShapeAndFlatVector(t: TensorLike): [Shape, FlatVector] {
  if ((t as BasicTensor).getData) {
    t = t as BasicTensor;
    return [t.shape, t.getData()];
  } else if (isTypedArray(t)) {
    return [[t.length], t];
  } else if (t instanceof Array) {
    return [inferShape(t), flatten(t) as number[]];
  } else if (typeof t == "number") {
    return [[], [t]];
  }
}

function toNumber(t: TensorLike): number {
  const [shape, values] = toShapeAndFlatVector(t);
  if (values.length != 1) {
    throw new Error("Not Scalar");
  }
  return values[0];
}

export function log(...args: any[]) {
  if (debug) {
    console.log.apply(null, args);
  }
}

export function shapesEqual(x: Shape, y: Shape): boolean {
  if (x.length != y.length) return false;
  for (let i = 0; i < x.length; ++i) {
    if (x[i] != y[i]) return false;
  }
  return true;
}

export function assert(expr: boolean, msg = "") {
  if (!expr) {
    throw new Error(msg);
  }
}

export function assertFalse(expr: boolean, msg = "") {
  assert(!expr, msg);
}

export function assertClose(actual: TensorLike, expected: TensorLike,
                            delta = 0.001) {
  actual = toNumber(actual);
  expected = toNumber(expected);
  assert(Math.abs(actual - expected) < delta,
    `actual: ${actual} expected: ${expected}`);
}

export function assertEqual(actual: TensorLike, expected: number|boolean,
                            msg = null) {
  actual = toNumber(actual);
  if (!msg) { msg = `actual: ${actual} expected: ${expected}`; }
  assert(actual === expected, msg);
}

export function assertShapesEqual(actual: Shape, expected: Shape) {
  const msg = `Shape mismatch. actual: ${J(actual)} expected ${J(expected)}`;
  assert(shapesEqual(actual, expected), msg);
}

export function assertAllEqual(actual: TensorLike, expected: TensorLike) {
  const [actualShape, actualFlat] = toShapeAndFlatVector(actual);
  const [expectedShape, expectedFlat] = toShapeAndFlatVector(expected);
  assertShapesEqual(actualShape, expectedShape);
  for (let i = 0; i < actualFlat.length; i++) {
    assert(actualFlat[i] === expectedFlat[i],
      `index ${i} actual: ${actualFlat[i]} expected: ${expectedFlat[i]}`);
  }
}

export function assertAllClose(actual: TensorLike, expected: TensorLike,
                               delta = 0.001) {
  const [actualShape, actualFlat] = toShapeAndFlatVector(actual);
  const [expectedShape, expectedFlat] = toShapeAndFlatVector(expected);

  assertShapesEqual(actualShape, expectedShape);

  for (let i = 0; i < actualFlat.length; ++i) {
    const a = (actualFlat[i]) as number;
    const e = (expectedFlat[i]) as number;
    assert(Math.abs(a - e) < delta,
      `index ${i} actual: ${actualFlat[i]} expected: ${expectedFlat[i]}`);
  }
}

// Provides a map with default value 0.
export class CounterMap {
  private map = new Map<number, number>();

  get(id: number): number {
    return this.map.has(id) ? this.map.get(id) : 0;
  }

  keys(): number[] {
    return Array.from(this.map.keys());

  }

  inc(id: number): void {
    this.map.set(id, this.get(id) + 1);
  }

  dec(id: number): void {
    this.map.set(id, this.get(id) - 1);
  }
}

