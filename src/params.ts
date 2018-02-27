import { Tensor } from "./api";
import { watch } from "./backprop";

/** Constructs a new params object.
 * Same as `new Params()`. See the documentation for in the Params class for
 * more info.
 *
 *    import * as pr from "propel";
 *    let params = pr.params();
 *    params.define("Weights", () => pr.randn([2, 5]));
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
 * 2) the initial value.
 *
 * The way this is done is with the define() method
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
   *    params.define("A", () => pr.randn([2]));
   *    params.define("B", () => pr.zeros([2, 2]));
   *    for (let [name, tensor] of params) {
   *      console.log(name);
   *      console.log(tensor);
   *    }
   */
  [Symbol.iterator]();

  /** Returns a subset of the Params with only the tensors that have the given
   * prefix.
   */
  scope(prefix: string): Params;

  /** Initializes a new tensor if it doesn't already exist in the
   * params object, otherwise returns the existing param of the given name.
   * All tensors in the params object get automatically traced for backprop.
   */
  define(name: string, initFn: () => Tensor): Tensor;
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

  [Symbol.iterator]() {
    return this.store[Symbol.iterator]();
  }

  /** Returns a subset of the Params with only the tensors that have the given
   * prefix.
   */
  scope(prefix: string): Params {
    return new ScopedParams(this, prefix);
  }

  define(name: string, initFn: () => Tensor): Tensor {
    let t = this.get(name);
    if (!t) {
      t = initFn();
      this.set(name, t);
      watch(t);
    }
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

  [Symbol.iterator]() {
    throw Error("implement me");
  }

  scope(prefix: string): Params {
    return this.parent.scope(this.resolve(prefix));
  }

  define(name: string, initFn: () => Tensor): Tensor {
    return this.parent.define(this.resolve(name), initFn);
  }
}
