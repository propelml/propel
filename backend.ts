// BasicTensor abstracts TensorFlow and DeepLearn BackendOps. Operations on
// BasicTensors are not traced in backprop and the class is not exposed to the
// public API.
import { flatten, inferShape } from "./deps/deeplearnjs/src/util";
import { OpsDL, TensorDL } from "./dl";
import { binding, OpsTF, TensorTF } from "./tf";
import * as types from "./types";
import { deepCloneArray } from "./util";

let tensorClass: any;
export let backend: string;
export let bo: types.BackendOps;
if (binding) {
  tensorClass = TensorTF;
  bo = new OpsTF();
  backend = "tf";
} else {
  tensorClass = TensorDL;
  bo = new OpsDL();
  backend = "dl";
}

function create(data: types.TypedArray, shape: types.Shape,
    dtype: types.DType): types.BasicTensor {
  const t = tensorClass.fromTypedArray(data, shape, dtype);
  return t;
}

export function convertBasic(x: types.TensorLike,
                             dtype?: types.DType): types.BasicTensor {
  if (typeof x === "number") {
    return create(types.makeTypedArray([x], dtype), [], dtype);
  } else if (types.isTypedArray(x)) {
    return create(x, [x.length], dtype);
  } else if (Array.isArray(x)) {
    if (!(x instanceof Array)) {
      // Unfortunately deeplearnjs gets confused by an out-of-context array.
      // Therefore clone the array.
      x = deepCloneArray(x);
    }
    const shape = inferShape(x);
    const data = flatten(x) as number[];
    return create(types.makeTypedArray(data, dtype), shape, dtype);
  }
  throw new Error("Unreachable");
}
