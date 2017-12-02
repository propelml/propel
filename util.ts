import { Shape, Tensor, TensorLike } from "./tensor";
import { FlatVector, flatten, inferShape } from "./deeplearnjs/src/util";

const debug = false;
const J = JSON.stringify;


function toShapeAndFlatVector(t: TensorLike): [Shape, FlatVector] {
  if (t instanceof Tensor) {
    return [t.shape, t.getValues()];
  } else if (t instanceof Array) {
    return [inferShape(t), flatten(t) as number[]];
  } else if (typeof t == "number") {
    return [[], [t]];
  } else if (typeof t == "boolean") {
    return [[], [t]];
  } else {
    // TypedArray
    return [[t.length], t];
  }
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
  actual = Tensor.convert(actual).toNumber();
  expected = Tensor.convert(expected).toNumber();
  assert(Math.abs(actual - expected) < delta,
    `actual: ${actual} expected: ${expected}`);
}

export function assertEqual(actual: TensorLike, expected: number|boolean,
                            msg = null) {
  actual = Tensor.convert(actual).toNumber();
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

export class GradientCollector {
  // Maps tensor id -> gradient tensor array
  private map = new Map<number, Tensor[]>();

  append(tid: number, grad: Tensor): void {
    if (this.map.has(tid)) {
      this.map.get(tid).push(grad);
    } else {
      this.map.set(tid, [grad]);
    }
  }

  // Sum up the gradients for a given tensor id.
  aggregate(tid: number): Tensor {
    if (!this.map.has(tid) || this.map.get(tid).length == 0) {
      // TODO(scalar) Handle non-scalar shapes.
      return Tensor.convert(0);
    }
    const grads = this.map.get(tid);
    //log('aggregate tid %d ngrads %d', tid, grads.length);
    let sum = grads[0];
    for (let i = 1; i < grads.length; i++) {
      sum = sum.add(grads[i]);
    }
    return sum;
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
