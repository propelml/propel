import {NDArrayMath} from './deeplearnjs/src/math/math'
import {NDArray} from './deeplearnjs/src/math/ndarray'
export {NDArray} from './deeplearnjs/src/math/ndarray'
import {NDArrayMathCPU} from './deeplearnjs/src/math/math_cpu';

export type TensorLike = number | number[] | NDArray | Tensor;

export class Tensor {
  private static nextId: number = 1;
  id: number;
  ndarray: NDArray;

  static ids(tensors: Tensor[]): number[] {
    return tensors.map(t => t.id);
  }

  static convert(x: TensorLike): Tensor {
    if (x instanceof Tensor) {
      return x;
    } else {
      return new Tensor(x);
    }
  }

  constructor(x: TensorLike) {
    if (typeof x == "number") {
      this.ndarray = NDArray.make([], {values: new Float32Array([x])});
    } else if (x instanceof Array) {
      this.ndarray = NDArray.make([], {values: new Float32Array(x)});
    } else if (x instanceof NDArray) {
      this.ndarray = x;
    }

    this.id = Tensor.nextId;
    Tensor.nextId++;
  }

  toNumber(): number {
    // TODO(scalar) 
    return this.ndarray.getValues()[0];
  }

  zerosLike(): Tensor {
    let zeros = NDArray.zerosLike(this.ndarray);
    return new Tensor(zeros);
  }

  onesLike(): Tensor {
    let ones = NDArray.zerosLike(this.ndarray);
    ones.fill(1.0);
    return new Tensor(ones);
  }

  toString(): string {
    // TODO(scalar) 
    return `[Tensor ${this.id} ${this.ndarray.getValues()}]`
  }
}
