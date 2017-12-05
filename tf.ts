// TensorFlow backend.
import { assertEqual } from "./util"; 
import * as types from "./types";

export function maybeRequireBinding() {
  // If we're in the browser, don't even attempt it.
  if (typeof window !== 'undefined') return null;

  // This is to set the backend to either web or tensorflow.
  // Use this on the command line:
  //   PROPEL=web node myprogram.js
  // This is used in tools/presubmit.sh to run tests on both backends through
  // node.
  const opts = (process.env.PROPEL || "").split(",");
  if (opts.indexOf("web") >= 0) return null;

  // Now require the compiled tensorflow-binding.node
  // When using ts-node, we are in the root dir, after compiling to
  // javascript, we are in the dist dir.
  const toAttempt = [
    '../../build/Debug/tensorflow-binding.node',
    '../../build/Release/tensorflow-binding.node',
    '../build/Debug/tensorflow-binding.node',
    '../build/Release/tensorflow-binding.node',
    './build/Debug/tensorflow-binding.node',
    './build/Release/tensorflow-binding.node',
  ];
  const fs = require("fs");
  const path = require("path");
  for (const fn of toAttempt) {
    if (fs.existsSync(path.join(__dirname, fn))) {
      return require(fn);
    }
  }
  console.log("TF binding not found. Falling back to DLJS.");
  return null;
}

export let binding = maybeRequireBinding();

// Auto create context for now.
export let ctx;
if (binding) {
  ctx = new binding.Context();
}

// Sugar for single value ops.
export function execute0(opName, inputs, attrs) {
  const r = binding.execute(ctx, opName, attrs, inputs);
  assertEqual(r.length, 1);
  return r[0];
}

function handleDType(handle): types.DType {
  switch (handle.dtype) {
    case binding.TF_FLOAT:
      return "float32";
    case binding.TF_INT32:
      return "int32";
    case binding.TF_UINT8:
      return "uint8";
    default:
      throw new Error("Not Implemented");
  }
}

export class BasicTensorTF implements types.BasicTensor {
  readonly dtype: types.DType;
  readonly shape: types.Shape;
  readonly handle: any;  // binding.Tensor
  private data?: types.TypedArray;

  static fromTypedArray(data: types.TypedArray, shape: types.Shape):
    BasicTensorTF {
    return new BasicTensorTF(new binding.Tensor(data, shape));
  }

  constructor(handle: any) {
    this.handle = handle;
    this.shape = handle.shape;
    this.dtype = handleDType(handle);
  }

  getData(): types.TypedArray {
    if (!this.data) {
      const ab = this.handle.asArrayBuffer();
      switch (this.dtype) {
        case "float32":
          this.data = new Float32Array(ab);
          break;
        case "int32":
          this.data = new Int32Array(ab);
          break;
        case "uint8":
          this.data = new Uint8Array(ab);
          break;
        case "bool":
          this.data = new Uint8Array(ab);
          break;
      }
    }
    return this.data;
  }
}

export class BasicOpsTF implements types.BasicOps {

  add(x: BasicTensorTF, y: BasicTensorTF): BasicTensorTF {
    const r = execute0("Add", [x.handle, y.handle], [
      ["T", binding.ATTR_TYPE, x.handle.dtype],
    ]);
    return new BasicTensorTF(r);
  }

  sub(x: BasicTensorTF, y: BasicTensorTF): BasicTensorTF {
    const r = execute0("Sub", [x.handle, y.handle], [
      ["T", binding.ATTR_TYPE, x.handle.dtype],
    ]);
    return new BasicTensorTF(r);
  }

  mul(x: BasicTensorTF, y: BasicTensorTF): BasicTensorTF {
    const r = execute0("Mul", [x.handle, y.handle], [
      ["T", binding.ATTR_TYPE, x.handle.dtype],
    ]);
    return new BasicTensorTF(r);
  }

  div(x: BasicTensorTF, y: BasicTensorTF): BasicTensorTF {
    const r = execute0("Div", [x.handle, y.handle], [
      ["T", binding.ATTR_TYPE, x.handle.dtype],
    ]);
    return new BasicTensorTF(r);
  }

  neg(x: BasicTensorTF): BasicTensorTF {
    const r = execute0("Neg", [x.handle], [
      ["T", binding.ATTR_TYPE, x.handle.dtype],
    ]);
    return new BasicTensorTF(r);
  }

  exp(x: BasicTensorTF): BasicTensorTF {
    const r = execute0("Exp", [x.handle], [
      ["T", binding.ATTR_TYPE, x.handle.dtype],
    ]);
    return new BasicTensorTF(r);
  }

  eye(size: number, dtype: types.DType = "float32"): types.BasicTensor {
    throw new Error("Not Implemented");
  }

  onesLike(x: BasicTensorTF): BasicTensorTF {
    const r = execute0("OnesLike", [x.handle], [
      ["T", binding.ATTR_TYPE, x.handle.dtype],
    ]);
    return new BasicTensorTF(r);
  }
}
