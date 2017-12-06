// BasicTensor abstracts TensorFlow and DeepLearn basicOps. Operations on
// BasicTensors are not traced in backprop and the class is not exposed to the
// public API.
import * as types from "./types";
import { flatten, inferShape } from "./deps/deeplearnjs/src/util";
import { BasicTensorDL, BasicOpsDL } from "./dl";
import { binding, BasicTensorTF, BasicOpsTF } from "./tf";

let tensorClass: any;
export let basicOps: types.BasicOps;
if (binding) {
  tensorClass = BasicTensorTF;
  basicOps = new BasicOpsTF();
} else {
  tensorClass = BasicTensorDL;
  basicOps = new BasicOpsDL();
}

function create(data: types.TypedArray, shape: types.Shape): types.BasicTensor {
  const t = tensorClass.fromTypedArray(data, shape);
  return t;
}

function makeTypedArray(data, dtype: types.DType): types.TypedArray {
  switch (dtype) {
    case "float32":
      return new Float32Array(data);
    case "int32":
      return new Int32Array(data);
    default:
      throw new Error("Not implemented");
  }
}

export function convertBasic(x: types.TensorLike,
  dtype: types.DType = "float32"): types.BasicTensor {
  if (typeof x == "number") {
    return create(makeTypedArray([x], dtype), []);
  } else if (types.isTypedArray(x)) {
    return create(x, [x.length]);
  } else if (x instanceof Array) {
    const shape = inferShape(x);
    const data = flatten(x) as number[];
    return create(makeTypedArray(data, dtype), shape);
  }
  throw new Error("Unreachable");
}

