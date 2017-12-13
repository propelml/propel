// TensorFlow backend.
import { convertBasic } from "./backend";
import { BindingInterface } from "./binding";
import * as types from "./types";
import { assert, assertEqual } from "./util";

function maybeRequireBinding(): BindingInterface | null {
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

function int32Scalar(v: number): TensorTF {
  return TensorTF.fromTypedArray(new Int32Array([v]), []);
}

function floatScalar(v: number): TensorTF {
  return TensorTF.fromTypedArray(new Float32Array([v]), []);
}

export class TensorTF implements types.BasicTensor {
  readonly dtype: types.DType;
  readonly shape: types.Shape;
  readonly handle: any;  // binding.Handle
  private data?: types.TypedArray;

  static fromTypedArray(data: types.TypedArray, shape: types.Shape,
                        dtype?: types.DType): TensorTF {
    if (dtype === undefined) {
      dtype = types.getDType(data);
    }
    const dtypeTF = dtypePropel2TF(dtype);
    return new TensorTF(new binding.Handle(data, shape, dtypeTF));
  }

  constructor(handle: any) {
    this.handle = handle;
    this.shape = binding.getShape(handle);
    this.dtype = dtypeTF2Propel(binding.getDType(handle));
  }

  getData(): types.TypedArray {
    if (!this.data) {
      const ab = binding.asArrayBuffer(this.handle);
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

export class OpsTF implements types.BackendOps {

  add(x: TensorTF, y: TensorTF): TensorTF {
    const r = execute0("Add", [x.handle, y.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
    ]);
    return new TensorTF(r);
  }

  sub(x: TensorTF, y: TensorTF): TensorTF {
    const r = execute0("Sub", [x.handle, y.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
    ]);
    return new TensorTF(r);
  }

  mul(x: TensorTF, y: TensorTF): TensorTF {
    const r = execute0("Mul", [x.handle, y.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
    ]);
    return new TensorTF(r);
  }

  div(x: TensorTF, y: TensorTF): TensorTF {
    const r = execute0("Div", [x.handle, y.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
    ]);
    return new TensorTF(r);
  }

  neg(x: TensorTF): TensorTF {
    const r = execute0("Neg", [x.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
    ]);
    return new TensorTF(r);
  }

  exp(x: TensorTF): TensorTF {
    const r = execute0("Exp", [x.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
    ]);
    return new TensorTF(r);
  }

  log(x: TensorTF): TensorTF {
    const r = execute0("Log", [x.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
    ]);
    return new TensorTF(r);
  }

  eye(size: number, dtype: types.DType = "float32"): types.BasicTensor {
    throw new Error("Not Implemented");
  }

  onesLike(x: TensorTF): TensorTF {
    const r = execute0("OnesLike", [x.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
    ]);
    return new TensorTF(r);
  }

  zerosLike(x: TensorTF): TensorTF {
    const r = execute0("ZerosLike", [x.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
    ]);
    return new TensorTF(r);
  }

  fill(value: TensorTF, shape: types.Shape): TensorTF {
    if (value.shape.length !== 0) {
      throw new Error("Fill value must be a scalar.");
    }
    const dims = TensorTF.fromTypedArray(new Int32Array(shape),
      [shape.length]);
    const r = execute0("Fill", [dims.handle, value.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(value.handle)],
    ]);
    return new TensorTF(r);
  }

  square(x: TensorTF): TensorTF {
    const r = execute0("Square", [x.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
    ]);
    return new TensorTF(r);
  }

  sinh(x: TensorTF): TensorTF {
    const r = execute0("Sinh", [x.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
    ]);
    return new TensorTF(r);
  }

  cosh(x: TensorTF): TensorTF {
    const r = execute0("Cosh", [x.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
    ]);
    return new TensorTF(r);
  }

  tanh(x: TensorTF): TensorTF {
    const r = execute0("Tanh", [x.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
    ]);
    return new TensorTF(r);
  }

  randn(shape: types.Shape, seed?: number): TensorTF {
    const shapeT = TensorTF.fromTypedArray(new Int32Array(shape),
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
    return new TensorTF(r);
  }

  linspace(start: number, stop: number, num: number): TensorTF {
    const startT = floatScalar(start);
    const stopT = floatScalar(stop);
    const numT = int32Scalar(num);
    const args = [startT.handle, stopT.handle, numT.handle];
    const r = execute0("LinSpace", args, [
      ["T", binding.ATTR_TYPE, binding.TF_FLOAT],
      ["Tidx", binding.ATTR_TYPE, binding.TF_INT32],
    ]);
    return new TensorTF(r);
  }

  arange(start: number, limit: number, delta: number): TensorTF {
    const startT = int32Scalar(start);
    const limitT = int32Scalar(limit);
    const deltaT = int32Scalar(delta);
    const args = [startT.handle, limitT.handle, deltaT.handle];
    const r = execute0("Range", args, [
      ["Tidx", binding.ATTR_TYPE, binding.TF_INT32],
    ]);
    return new TensorTF(r);
  }

  transpose(x: TensorTF, perm: TensorTF): TensorTF {
    const r = execute0("Transpose", [x.handle, perm.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
      ["Tperm", binding.ATTR_TYPE, binding.getDType(perm.handle)],
    ]);
    return new TensorTF(r);
  }

  reverse(x: TensorTF, dims: TensorTF): TensorTF {
    assert(dims.dtype === "bool");
    const r = execute0("Reverse", [x.handle, dims.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
    ]);
    return new TensorTF(r);
  }

  matmul(x: TensorTF, y: TensorTF, transposeA = false,
         transposeB = false): TensorTF {
    const r = execute0("MatMul", [x.handle, y.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
      ["transpose_a", binding.ATTR_BOOL, transposeA],
      ["transpose_b", binding.ATTR_BOOL, transposeB],
    ]);
    return new TensorTF(r);
  }

  argmax(x: TensorTF, axis: number): TensorTF {
    const axisT = int32Scalar(axis);
    const r = execute0("ArgMax", [x.handle, axisT.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
      ["Tidx", binding.ATTR_TYPE, binding.TF_INT32],
      ["output_type", binding.ATTR_TYPE, binding.TF_INT32],
    ]);
    return new TensorTF(r);
  }

  argmin(x: TensorTF, axis: number): TensorTF {
    const axisT = int32Scalar(axis);
    const r = execute0("ArgMin", [x.handle, axisT.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
      ["Tidx", binding.ATTR_TYPE, binding.TF_INT32],
      ["output_type", binding.ATTR_TYPE, binding.TF_INT32],
    ]);
    return new TensorTF(r);
  }

  reduceSum(x: TensorTF, axes: number[], keepDims: boolean): TensorTF
  {
    const axesT = convertBasic(axes, "int32") as TensorTF;
    const r = execute0("Sum", [x.handle, axesT.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
      ["Tidx", binding.ATTR_TYPE, binding.TF_INT32],
      ["keep_dims", binding.ATTR_BOOL, keepDims],
    ]);
    return new TensorTF(r);
  }

  reduceMax(x: TensorTF, axes: number[], keepDims: boolean): TensorTF
  {
    const axesT = convertBasic(axes, "int32") as TensorTF;
    const r = execute0("Max", [x.handle, axesT.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
      ["Tidx", binding.ATTR_TYPE, binding.TF_INT32],
      ["keep_dims", binding.ATTR_BOOL, keepDims],
    ]);
    return new TensorTF(r);
  }

  equal(x: TensorTF, y: TensorTF): TensorTF {
    const r = execute0("Equal", [x.handle, y.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
    ]);
    return new TensorTF(r);
  }

  slice(x: TensorTF, begin: number[], size: number[]): TensorTF {
    const beginT = convertBasic(begin, "int32") as TensorTF;
    const sizeT = convertBasic(size, "int32") as TensorTF;
    const r = execute0("Slice", [x.handle, beginT.handle, sizeT.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
      ["Index", binding.ATTR_TYPE, binding.TF_INT32],
    ]);
    return new TensorTF(r);
  }

  reshape(x: TensorTF, newShape: types.Shape): TensorTF {
    const shapeT = convertBasic(newShape, "int32") as TensorTF;
    const r = execute0("Reshape", [x.handle, shapeT.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
      ["Tshape", binding.ATTR_TYPE, binding.getDType(shapeT.handle)],
    ]);
    return new TensorTF(r);
  }

  softmax(x: TensorTF): TensorTF {
    const r = execute0("Softmax", [x.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
    ]);
    return new TensorTF(r);
  }

  logSoftmax(x: TensorTF): TensorTF {
    const r = execute0("LogSoftmax", [x.handle], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
    ]);
    return new TensorTF(r);
  }

  cast(x: TensorTF, dtype: types.DType): TensorTF {
    const r = execute0("Cast", [x.handle], [
      ["SrcT", binding.ATTR_TYPE, binding.getDType(x.handle)],
      ["DstT", binding.ATTR_TYPE, dtypePropel2TF(dtype)],
    ]);
    return new TensorTF(r);
  }

  oneHot(x: TensorTF, depth: number, onValue: number,
         offValue: number): TensorTF {
    if (x.dtype === "float32") {
      throw new Error("Must use integer type with oneHot.");
    }
    const depthT = int32Scalar(depth);
    const onT = floatScalar(onValue);
    const offT = floatScalar(offValue);
    const args = [x.handle, depthT.handle, onT.handle, offT.handle];
    const r = execute0("OneHot", args, [
      ["T", binding.ATTR_TYPE, binding.getDType(onT.handle)],
      ["TI", binding.ATTR_TYPE, binding.getDType(x.handle)],
      ["axis", binding.ATTR_INT, -1],
    ]);
    return new TensorTF(r);
  }
}
