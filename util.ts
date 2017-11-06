import {Tensor, TensorLike} from "./tensor";

let debug = false;

export function log(...args: any[]) {
  if (debug) {
    console.log.apply(null, args);
  }
}

export function assert(expr: boolean, msg="") {
  if (!expr) {
    throw new Error(msg);
  }
}

export function assertClose(actual: TensorLike, expected: TensorLike) {
  actual = Tensor.convert(actual).toNumber();
  expected = Tensor.convert(expected).toNumber();
  assert(Math.abs(actual - expected) < 0.001,
         `actual: ${actual} expected: ${expected}`);
}

export function assertEqual(actual: TensorLike, expected: number) {
  actual = Tensor.convert(actual).toNumber();
  assert(actual == expected, `actual: ${actual} expected: ${expected}`);
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
    let grads = this.map.get(tid);
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
