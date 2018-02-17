import { Tensor, randn, zeros } from "./api";
import * as types from "./types";
import { bo } from "./backend";

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
 *
 * Iterate over it like this:
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
export class Params {
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

  /** If the given name does not exist in the parameters object, this
   * initializes a new random normal tensor. If the name does exist
   * in the parameters object, this just returns that stored tensor.
   */
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
