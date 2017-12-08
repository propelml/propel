// TensorFlow backend.
import { convertBasic } from "./basic";
import * as types from "./types";
import { assert, assertEqual } from "./util";

export function maybeRequireBinding() {
  // If we're in the browser, don't even attempt it.
  if (typeof window !== "undefined") return null;

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
    "../../build/Debug/tensorflow-binding.node",
    "../../build/Release/tensorflow-binding.node",
    "../build/Debug/tensorflow-binding.node",
    "../build/Release/tensorflow-binding.node",
    "./build/Debug/tensorflow-binding.node",
    "./build/Release/tensorflow-binding.node",
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

function dtypeTF2Propel(dtypeTF: number): types.DType {
  switch (dtypeTF) {
    case binding.TF_BOOL:
      return "bool";
    case binding.TF_FLOAT:
      return "float32";
    case binding.TF_INT32:
      return "int32";
    case binding.TF_UINT8:
      return "uint8";
    default:
      throw new Error(`Not Implemented: dtype ${dtypeTF}`);
  }
}

function dtypePropel2TF(dtype: types.DType): number {
  switch (dtype) {
    case "bool":
      return binding.TF_BOOL;
    case "float32":
      return binding.TF_FLOAT;
    case "int32":
      return binding.TF_INT32;
    case "uint8":
      return binding.TF_UINT8;
    default:
      throw new Error(`Not Implemented`);
  }
}

function int32Scalar(v: number): BasicTensorTF {
  return BasicTensorTF.fromTypedArray(new Int32Array([v]), []);
}

function floatScalar(v: number): BasicTensorTF {
  return BasicTensorTF.fromTypedArray(new Float32Array([v]), []);
}

export class BasicTensorTF implements types.BasicTensor {
  readonly dtype: types.DType;
  readonly shape: types.Shape;
  readonly handle: any;  // binding.Tensor
  private data?: types.TypedArray;

  static fromTypedArray(data: types.TypedArray, shape: types.Shape,
                        dtype?: types.DType): BasicTensorTF {
    if (dtype === undefined) {
      dtype = types.getDType(data);
    }
    const dtypeTF = dtypePropel2TF(dtype);
    return new BasicTensorTF(new binding.Tensor(data, shape, dtypeTF));
  }

  constructor(handle: any) {
    this.handle = handle;
    this.shape = handle.shape;
    this.dtype = dtypeTF2Propel(handle.dtype);
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

  square(x: BasicTensorTF): BasicTensorTF {
    const r = execute0("Square", [x.handle], [
      ["T", binding.ATTR_TYPE, x.handle.dtype],
    ]);
    return new BasicTensorTF(r);
  }

  sinh(x: BasicTensorTF): BasicTensorTF {
    const r = execute0("Sinh", [x.handle], [
      ["T", binding.ATTR_TYPE, x.handle.dtype],
    ]);
    return new BasicTensorTF(r);
  }

  cosh(x: BasicTensorTF): BasicTensorTF {
    const r = execute0("Cosh", [x.handle], [
      ["T", binding.ATTR_TYPE, x.handle.dtype],
    ]);
    return new BasicTensorTF(r);
  }

  tanh(x: BasicTensorTF): BasicTensorTF {
    const r = execute0("Tanh", [x.handle], [
      ["T", binding.ATTR_TYPE, x.handle.dtype],
    ]);
    return new BasicTensorTF(r);
  }

  randn(shape: types.Shape, seed?: number): BasicTensorTF {
    const shapeT = BasicTensorTF.fromTypedArray(new Int32Array(shape),
      [shape.length]);
    const args = [shapeT.handle];
    if (typeof seed !== "number") seed = 0;
    const attrs = [
      ["dtype", binding.ATTR_TYPE, binding.TF_FLOAT], // output
      ["T", binding.ATTR_TYPE, binding.TF_INT32], // shape
      ["seed", binding.ATTR_INT, seed],
      ["seed2", binding.ATTR_INT, seed],
    ];
    const r = execute0("RandomStandardNormal", args, attrs);
    return new BasicTensorTF(r);
  }

  linspace(start: number, stop: number, num: number): BasicTensorTF {
    const startT = floatScalar(start);
    const stopT = floatScalar(stop);
    const numT = int32Scalar(num);
    const args = [startT.handle, stopT.handle, numT.handle];
    const r = execute0("LinSpace", args, [
      ["T", binding.ATTR_TYPE, binding.TF_FLOAT],
      ["Tidx", binding.ATTR_TYPE, binding.TF_INT32],
    ]);
    return new BasicTensorTF(r);
  }

  arange(start: number, limit: number, delta: number): BasicTensorTF {
    const startT = int32Scalar(start);
    const limitT = int32Scalar(limit);
    const deltaT = int32Scalar(delta);
    const args = [startT.handle, limitT.handle, deltaT.handle];
    const r = execute0("Range", args, [
      ["Tidx", binding.ATTR_TYPE, binding.TF_INT32],
    ]);
    return new BasicTensorTF(r);
  }

  transpose(x: BasicTensorTF, perm: BasicTensorTF): BasicTensorTF {
    const r = execute0("Transpose", [x.handle, perm.handle], [
      ["T", binding.ATTR_TYPE, x.handle.dtype],
      ["Tperm", binding.ATTR_TYPE, perm.handle.dtype],
    ]);
    return new BasicTensorTF(r);
  }

  reverse(x: BasicTensorTF, dims: BasicTensorTF): BasicTensorTF {
    assert(dims.dtype === "bool");
    const r = execute0("Reverse", [x.handle, dims.handle], [
      ["T", binding.ATTR_TYPE, x.handle.dtype],
    ]);
    return new BasicTensorTF(r);
  }

  matmul(x: BasicTensorTF, y: BasicTensorTF, transposeA = false,
         transposeB = false): BasicTensorTF {
    const r = execute0("MatMul", [x.handle, y.handle], [
      ["T", binding.ATTR_TYPE, x.handle.dtype],
      ["transpose_a", binding.ATTR_BOOL, transposeA],
      ["transpose_b", binding.ATTR_BOOL, transposeB],
    ]);
    return new BasicTensorTF(r);
  }

  reduceSum(x: BasicTensorTF, axes: number[], keepDims: boolean): BasicTensorTF
  {
    const axesT = convertBasic(axes, "int32") as BasicTensorTF;
    const r = execute0("Sum", [x.handle, axesT.handle], [
      ["T", binding.ATTR_TYPE, x.handle.dtype],
      ["Tidx", binding.ATTR_TYPE, binding.TF_INT32],
      ["keep_dims", binding.ATTR_BOOL, keepDims],
    ]);
    return new BasicTensorTF(r);
  }

  reduceMax(x: BasicTensorTF, axes: number[], keepDims: boolean): BasicTensorTF
  {
    const axesT = convertBasic(axes, "int32") as BasicTensorTF;
    const r = execute0("Max", [x.handle, axesT.handle], [
      ["T", binding.ATTR_TYPE, x.handle.dtype],
      ["Tidx", binding.ATTR_TYPE, binding.TF_INT32],
      ["keep_dims", binding.ATTR_BOOL, keepDims],
    ]);
    return new BasicTensorTF(r);
  }
}
