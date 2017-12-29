// BasicTensor abstracts TensorFlow and DeepLearn BackendOps. Operations on
// BasicTensors are not traced in backprop and the class is not exposed to the
// public API.
import { OpsDL, TensorDL } from "./dl";
import { binding, OpsTF, TensorTF } from "./tf";
import * as types from "./types";
import { deepCloneArray, flatten, inferShape } from "./util";

let tensorClass: any;
export let backend: string;
export let bo: types.BackendOps;
if (binding) {
  console.log("Using TF backend.");
  tensorClass = TensorTF;
  bo = new OpsTF();
  backend = "tf";
} else {
  console.log("Using DL backend.");
  tensorClass = TensorDL;
  bo = new OpsDL();
  backend = "dl";
}

const create = tensorClass.fromTypedArray;

export function convertBasic(x: types.TensorLike,
    dtype?: types.DType, device?: string): types.BasicTensor {
  if (typeof x === "number") {
    // TODO On TF we should take advantage of createSmallHandle for scalars.
    return create(types.makeTypedArray([x], dtype), [], dtype, device);
  } else if (types.isTypedArray(x)) {
    return create(x, [x.length], dtype, device);
  } else if (Array.isArray(x)) {
    if (!(x instanceof Array)) {
      // Unfortunately deeplearnjs gets confused by an out-of-context array.
      // Therefore clone the array.
      x = deepCloneArray(x);
    }
    const shape = inferShape(x);
    const data = flatten(x) as number[];
    return create(types.makeTypedArray(data, dtype), shape, dtype, device);
  }
  throw new Error("Unreachable");
}
