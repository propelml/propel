import { randn, Tensor, zeros } from "./api";
import { bo } from "./backend";
import * as types from "./types";

/** Constructs a new params object.
 * Same as `new Params()`. See the documentation for in the Params class for
 * more info.
 *
 *    import * as pr from "propel";
 *    let params = pr.params();
 *    params.randn("Weights", [2, 5]);
 */
export function params(): Params {
  return new RootParams();
}

/** A collection of named Tensors.
 *
 * This is the primary object used to manage the "parameters" of neural
 * networks (sometimes called "weights" or "variables"). In Propel, the params
 * are pass explicitly to computations.
 *
 * To define a new parameter tensor, you must define
 * 1) a unique name
 * 2) the shape and dtype of the tensor
 * 3) the initial value.
 *
 * The way this is done is by methods randn() and zeros()
 * on the params object. They are each given a name, shape, and dtype,
 * if the name doesn't already exist in the params object, it is initialized.
 * Otherwise it is returned without modification.
 *
 * The params object is often used with optimizers to do gradient decent.
 *
 * If using saver(), the params object wont need to be constructed manually.
 */
export interface Params {
  has(name: string): boolean;
  get(name: string): Tensor;
  set(name: string, t: Tensor): Tensor;

  /** Iterates over the tensors in params;
   *
   *    import * as pr from "propel";
   *    let params = pr.params();
   *    params.randn("A", [2]);
   *    params.zeros("B", [2, 2]);
   *    params.forEach((tensor, name) => {
   *      console.log(name);
   *      console.log(tensor);
   *    });
   */
  forEach(cb: (t: Tensor, name: string) => void): void;

  /** Returns a subset of the Params with only the tensors that have the given
   * prefix.
   */
  scope(prefix: string): Params;

  // TODO The following  should have the same interface as top-level ones in
  // api.

  /** If the given name does not exist in the parameters object, this
   * initializes a new random normal tensor. If the name does exist
   * in the parameters object, this just returns that stored tensor.
   */
  randn(name: string, shape: types.Shape, opts?): Tensor;

  /** If the given name does not exist in the parameters object, this
   * initializes a new tensor with zero values. If the name does exist
   * in the parameters object, this just returns that stored tensor.
   */
  zeros(name: string, shape: types.Shape, dtype?: types.DType,
        device?: string): Tensor;
}

class RootParams implements Params {
  // Note TS doesn't allow extending Map:
  // https://github.com/Microsoft/TypeScript/issues/10853
  private store = new Map<string, Tensor>();

  has(name: string): boolean {
    return this.store.has(name);
  }

  get(name: string): Tensor {
    return this.store.get(name);
  }

  set(name: string, t: Tensor): Tensor {
    this.store.set(name, t);
    return t;
  }

  forEach(cb): void {
    this.store.forEach(cb);
  }

  /** Returns a subset of the Params with only the tensors that have the given
   * prefix.
   */
  scope(prefix: string): Params {
    return new ScopedParams(this, prefix);
  }

  randn(name: string, shape: types.Shape,
        { device = "CPU:0", scale = 0.1 } = {}): Tensor {
    if (!(shape instanceof Array)) {
      throw new Error("Randn takes a shape as an argument");
    }
    if (this.has(name)) {
      return this.get(name);
    }
    // Initialize.
    let t = randn(shape).mul(scale);
    if (device && device !== "CPU:0") {
      t = new Tensor(bo.copyToDevice(t.basic, device));
    }
    this.set(name, t);
    return t;
  }

  /** If the given name does not exist in the parameters object, this
   * initializes a new tensor with zero values. If the name does exist
   * in the parameters object, this just returns that stored tensor.
   */
  zeros(name: string, shape: types.Shape, dtype: types.DType = "float32",
        device = "CPU:0"): Tensor {
    if (!(shape instanceof Array)) {
      throw new Error("Zeros takes a shape as an argument");
    }
    if (this.has(name)) {
      return this.get(name);
    }
    // Initialize.
    let t = zeros(shape);
    if (device && device !== "CPU:0") {
      t = new Tensor(bo.copyToDevice(t.basic, device));
    }
    this.set(name, t);
    return t;
  }
}

class ScopedParams implements Params {
  constructor(readonly parent: Params, readonly prefix: string) { }

  private resolve(name: string): string {
    return this.prefix + "/" + name;
  }

  has(name: string): boolean {
    return this.parent.has(this.resolve(name));
  }

  get(name: string): Tensor {
    return this.parent.get(this.resolve(name));
  }

  set(name: string, t: Tensor): Tensor {
    return this.parent.set(this.resolve(name), t);
  }

  forEach(cb): void {
    this.parent.forEach((t: Tensor, name: string) => {
      if (name.startsWith(this.prefix)) {
        cb(t, name);
      }
    });
  }

  scope(prefix: string): Params {
    return this.parent.scope(this.resolve(prefix));
  }

  randn(name: string, shape: types.Shape,
        { device = "CPU:0", scale = 0.1 } = {}): Tensor {
    return this.parent.randn(this.resolve(name), shape, {device, scale});
  }

  zeros(name: string, shape: types.Shape, dtype: types.DType = "float32",
        device = "CPU:0"): Tensor {
    return this.parent.zeros(this.resolve(name), shape, dtype, device);
  }
}
