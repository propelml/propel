import {NDArray} from './deeplearnjs/src/math/ndarray'
export {NDArray} from './deeplearnjs/src/math/ndarray'
import * as ops from './ops';
import {RegularArray,inferShape,flatten} from './deeplearnjs/src/util';

export type TensorLike = number | RegularArray<number> | NDArray | Tensor;
type Shape = number[];

export class Tensor {
  private static nextId: number = 1;
  id: number;
  shape: Shape;
  ndarray: NDArray;
  dtype: 'float32' | 'uint8';

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
    if (x instanceof Array) {
      // Argument is a JS array like [[1, 2], [3, 4]].
      let shape = inferShape(x);
      let data = flatten(x) as Array<number>;
      this.ndarray = NDArray.make(shape, {values: new Float32Array(data)});
      this.shape = shape;
    } else if (typeof x == "number") {
      // Scalar
      this.ndarray = NDArray.make([], {values: new Float32Array([x])});
      this.shape = [1];
    } else if (x instanceof NDArray) {
      this.ndarray = x;
      this.shape = x.shape;
    }

    this.dtype = 'float32'; // TODO Support other dtypes.

    this.id = Tensor.nextId;
    Tensor.nextId++;
  }

  toNumber(): number {
    let values = this.ndarray.getValues();
    if (values.length != 1) {
      throw new Error("toNumber() can only be used on scalar tensors.");
    }
    return values[0];
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

  exp(): Tensor {
    return (new ops.Exp()).run(this);
  }

  neg(): Tensor {
    return (new ops.Neg()).run(this);
  }

  add(a: TensorLike): Tensor {
    a = Tensor.convert(a);
    return (new ops.Add()).run(this, a);
  }

  sub(a: TensorLike): Tensor {
    a = Tensor.convert(a);
    return (new ops.Sub()).run(this, a);
  }

  div(a: TensorLike): Tensor {
    a = Tensor.convert(a);
    return (new ops.Div()).run(this, a);
  }

  mul(a: TensorLike): Tensor {
    a = Tensor.convert(a);
    return (new ops.Mul()).run(this, a);
  }
}
