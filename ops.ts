import {Tensor} from "./tensor";
import * as backprop from './backprop';

export class Exp extends backprop.Op {
  input: Tensor;

  forward(a: Tensor): Tensor {
    this.input = a;
    return new Tensor(this.math.exp(a.ndarray));
  }

  backward(grad: Tensor): Tensor[] {
    let a = this.math.exp(this.input.ndarray);
    let g = this.math.multiply(grad.ndarray, a);
    return [new Tensor(g)];
  }
}

export class Neg extends backprop.Op {
  forward(a: Tensor): Tensor {
    return new Tensor(this.math.neg(a.ndarray));
  }

  backward(grad: Tensor): Tensor[] {
    return [this.forward(grad)];
  }
}

export class Add extends backprop.Op {
  forward(a: Tensor, b: Tensor): Tensor {
    let x = this.math.add(a.ndarray, b.ndarray);
    return new Tensor(x);
  }

  backward(grad: Tensor): Tensor[] {
    return [grad, grad];
  }
}

export class Sub extends backprop.Op {
  forward(a: Tensor, b: Tensor): Tensor {
    return new Tensor(this.math.sub(a.ndarray, b.ndarray));
  }

  backward(grad: Tensor): Tensor[] {
    let ga = grad;
    let gb = new Tensor(this.math.neg(grad.ndarray));
    return [ga, gb];
  }
}

export class Mul extends backprop.Op {
  inputs: Tensor[];

  forward(a: Tensor, b: Tensor): Tensor {
    this.inputs = [a, b];
    return new Tensor(this.math.multiply(a.ndarray, b.ndarray));
  }

  backward(grad: Tensor): Tensor[] {
    let ag = this.math.multiply(this.inputs[1].ndarray, grad.ndarray);
    let bg = this.math.multiply(this.inputs[0].ndarray, grad.ndarray);
    return [new Tensor(ag), new Tensor(bg)];
  }
}

export class Div extends backprop.Op {
  inputs: Tensor[];

  forward(a: Tensor, b: Tensor): Tensor {
    this.inputs = [a, b];
    return new Tensor(this.math.divide(a.ndarray, b.ndarray));
  }

  backward(grad: Tensor): Tensor[] {
    // f(a, b) = a / b 
    // df/da(a, b) = 1 / b 
    // df/db(a, b) = -a / b^2
    let a = this.inputs[0].ndarray;
    let b = this.inputs[1].ndarray;
    let ag = this.math.divide(grad.ndarray, b);
    let b2 = this.math.multiply(b, b);
    let bg = this.math.multiply(grad.ndarray, this.math.divide(this.math.neg(a), b2));
    return [new Tensor(ag), new Tensor(bg)];
  }
}
