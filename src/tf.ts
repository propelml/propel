/*!
   Copyright 2018 Propel http://propel.site/.  All rights reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */
// TensorFlow backend.
import { assert, assertEqual, getDType } from "./tensor_util";
import {
  AttrDef,
  DTypeCode,
  Handle,
} from "./tf_binding";
import * as types from "./types";

export let binding;
export let ctx;

export function loadBinding(): boolean {
  binding = require("./load_tf_binding");
  if (binding) {
    ctx = new binding.Context(); // Auto create context for now.
    return true;
  } else {
    return false;
  }
}

// Sugar for single value ops.
export function execute0(opName: string, inputs: TensorTF[], attrs): TensorTF {
  const handles = inputs.map((t) => t.handle);
  const r = binding.execute(ctx, opName, attrs, handles);
  assertEqual(r.length, 1);
  return new TensorTF(r[0]);
}

// Execute a simple op, which may have multiple inputs, but only a single
// attribute T, and only returns a single value.
// The returned tensor dtype will be same as first input, unless specified by
// the dtype argument.
export function execute1(opName: string, inputs: TensorTF[],
                         dtype?: types.DType): TensorTF {
  const handles = inputs.map((t) => t.handle);
  const dtypeTF = dtype == null ? binding.getDType(handles[0])
                                : dtypePropel2TF(dtype);
  const attrs = [["T", binding.ATTR_TYPE, dtypeTF]];
  const r = binding.execute(ctx, opName, attrs, handles);
  return new TensorTF(r[0]);
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
      throw new Error(`Not Implemented ${dtype}`);
  }
}

function colocateDevice(colocateWith?: TensorTF): string {
  return colocateWith ? binding.getDevice(colocateWith.handle) : "CPU:0";
}

function int32Small(v: number | number[], colocateWith?: TensorTF): TensorTF {
  return new TensorTF(binding.createSmallHandle(ctx, binding.TF_INT32,
    colocateDevice(colocateWith), v));
}

function floatSmall(v: number | number[], colocateWith?: TensorTF): TensorTF {
  return new TensorTF(binding.createSmallHandle(ctx, binding.TF_FLOAT,
    colocateDevice(colocateWith), v));
}

export class TensorTF implements types.Storage {
  handle: null | Handle;
  private data_?: types.TypedArray;

  constructor(handle: Handle) {
    this.handle = handle;
  }

  get shape(): types.Shape {
    return binding.getShape(this.handle);
  }

  get dtype(): types.DType {
    return dtypeTF2Propel(binding.getDType(this.handle));
  }

  get device(): string {
    return simplifyDeviceName(binding.getDevice(this.handle));
  }

  async data(): Promise<types.TypedArray> {
    // TODO maybe we can do better than this in TF. But TF is already pretty
    // fast, unlike webgl.
    return this.dataSync();
  }

  dataSync(): types.TypedArray {
    if (!this.data_) {
      const ab = binding.asArrayBuffer(this.handle);
      switch (this.dtype) {
        case "float32":
          this.data_ = new Float32Array(ab);
          break;
        case "int32":
          this.data_ = new Int32Array(ab);
          break;
        case "uint8":
          this.data_ = new Uint8Array(ab);
          break;
        case "bool":
          this.data_ = new Uint8Array(ab);
          break;
      }
    }
    return this.data_;
  }

  dispose(): void {
    assert(this.handle != null);
    binding.dispose(this.handle);
    this.handle = null;
  }
}

// TF has rather verbose device names like:
// '/job:localhost/replica:0/task:0/device:GPU:0'. Until Propel starts thinking
// about multi-replica configurations, we simplify this string to just "GPU:0".
function simplifyDeviceName(device: string): string {
  return device.split("/").pop().replace("device:", "");
}

export class OpsTF implements types.BackendOps {

  copyToDevice(x: TensorTF, device: string): TensorTF {
    const h = binding.copyToDevice(ctx, x.handle, device);
    return new TensorTF(h);
  }

  getDevice(x: TensorTF): string {
    return x.device;
  }

  listDevices(): string[] {
    return binding.listDevices(ctx).map(deviceDesc => {
      return simplifyDeviceName(deviceDesc.name);
    });
  }

  fromTypedArray(data: types.TypedArray, shape: types.Shape,
                 dtype?: types.DType, device?: string): TensorTF {
    if (dtype == null) {
      dtype = getDType(data);
    }
    const dtypeTF = dtypePropel2TF(dtype);
    let h = new binding.Handle(data, shape, dtypeTF);
    if (device && device !== "CPU:0") {
      h = binding.copyToDevice(ctx, h, device);
    }
    return new TensorTF(h);
  }

  add(x: TensorTF, y: TensorTF): TensorTF {
    return execute1("Add", [x, y]);
  }

  sub(x: TensorTF, y: TensorTF): TensorTF {
    return execute1("Sub", [x, y]);
  }

  mul(x: TensorTF, y: TensorTF): TensorTF {
    return execute1("Mul", [x, y]);
  }

  div(x: TensorTF, y: TensorTF): TensorTF {
    return execute1("Div", [x, y]);
  }

  neg(x: TensorTF): TensorTF {
    return execute1("Neg", [x]);
  }

  exp(x: TensorTF): TensorTF {
    return execute1("Exp", [x]);
  }

  log(x: TensorTF): TensorTF {
    return execute1("Log", [x]);
  }

  setDiag(input: TensorTF, diag: TensorTF): TensorTF {
    return execute1("MatrixSetDiag", [input, diag]);
  }

  onesLike(x: TensorTF): TensorTF {
    return execute1("OnesLike", [x]);
  }

  zerosLike(x: TensorTF): TensorTF {
    return execute1("ZerosLike", [x]);
  }

  fill(value: TensorTF, shape: types.Shape): TensorTF {
    if (value.shape.length !== 0) {
      throw new Error("Fill value must be a scalar.");
    }
    const shapeT = int32Small(shape);
    return execute1("Fill", [shapeT, value], value.dtype);
  }

  square(x: TensorTF): TensorTF {
    return execute1("Square", [x]);
  }

  sqrt(x: TensorTF): TensorTF {
    return execute1("Sqrt", [x]);
  }

  pow(x: TensorTF, exponent: TensorTF): TensorTF {
    return execute1("Pow", [x, exponent]);
  }

  sin(x: TensorTF): TensorTF {
    return execute1("Sin", [x]);
  }

  cos(x: TensorTF): TensorTF {
    return execute1("Cos", [x]);
  }

  tan(x: TensorTF): TensorTF {
    return execute1("Tan", [x]);
  }

  sinh(x: TensorTF): TensorTF {
    return execute1("Sinh", [x]);
  }

  cosh(x: TensorTF): TensorTF {
    return execute1("Cosh", [x]);
  }

  tanh(x: TensorTF): TensorTF {
    return execute1("Tanh", [x]);
  }

  relu(x: TensorTF): TensorTF {
    return execute1("Relu", [x]);
  }

  reluGrad(grad: TensorTF, features: TensorTF): TensorTF {
    return execute1("ReluGrad", [grad, features]);
  }

  sigmoid(x: TensorTF): TensorTF {
    return execute1("Sigmoid", [x]);
  }

  abs(x: TensorTF): TensorTF {
    return execute1("Abs", [x]);
  }

  randn(shape: types.Shape, seed?: number): TensorTF {
    const shapeT = int32Small(shape);
    if (typeof seed !== "number") seed = 0;
    const attrs = [
      ["dtype", binding.ATTR_TYPE, binding.TF_FLOAT], // output
      ["T", binding.ATTR_TYPE, binding.TF_INT32], // shape
      ["seed", binding.ATTR_INT, seed],
      ["seed2", binding.ATTR_INT, seed],
    ];
    return execute0("RandomStandardNormal", [shapeT], attrs);
  }

  linspace(start: number, stop: number, num: number): TensorTF {
    const startT = floatSmall(start);
    const stopT = floatSmall(stop);
    const numT = int32Small(num);
    return execute0("LinSpace", [startT, stopT, numT], [
      ["T", binding.ATTR_TYPE, binding.TF_FLOAT],
      ["Tidx", binding.ATTR_TYPE, binding.TF_INT32],
    ]);
  }

  range(start: number, limit: number, delta: number): TensorTF {
    const startT = int32Small(start);
    const limitT = int32Small(limit);
    const deltaT = int32Small(delta);
    const args = [startT, limitT, deltaT];
    return execute0("Range", args, [
      ["Tidx", binding.ATTR_TYPE, binding.TF_INT32],
    ]);
  }

  transpose(x: TensorTF, perm: TensorTF): TensorTF {
    return execute0("Transpose", [x, perm], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
      ["Tperm", binding.ATTR_TYPE, binding.getDType(perm.handle)],
    ]);
  }

  reverse(x: TensorTF, dims: TensorTF): TensorTF {
    assert(dims.dtype === "bool");
    return execute1("Reverse", [x, dims]);
  }

  matmul(x: TensorTF, y: TensorTF, transposeA = false,
         transposeB = false): TensorTF {
    return execute0("MatMul", [x, y], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
      ["transpose_a", binding.ATTR_BOOL, transposeA],
      ["transpose_b", binding.ATTR_BOOL, transposeB],
    ]);
  }

  argmax(x: TensorTF, axis: number): TensorTF {
    // axisT is expected to be on CPU.
    const axisT = int32Small(axis);
    return execute0("ArgMax", [x, axisT], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
      ["Tidx", binding.ATTR_TYPE, binding.TF_INT32],
      ["output_type", binding.ATTR_TYPE, binding.TF_INT32],
    ]);
  }

  argmin(x: TensorTF, axis: number): TensorTF {
    // axisT is expected to be on CPU.
    const axisT = int32Small(axis);
    return execute0("ArgMin", [x, axisT], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
      ["Tidx", binding.ATTR_TYPE, binding.TF_INT32],
      ["output_type", binding.ATTR_TYPE, binding.TF_INT32],
    ]);
  }

  reduceSum(x: TensorTF, axes: number[], keepDims: boolean): TensorTF
  {
    // axesT is expected to be on CPU.
    const axesT = int32Small(axes);
    return execute0("Sum", [x, axesT], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
      ["Tidx", binding.ATTR_TYPE, binding.TF_INT32],
      ["keep_dims", binding.ATTR_BOOL, keepDims],
    ]);
  }

  reduceMean(x: TensorTF, axes: number[], keepDims: boolean): TensorTF
  {
    // reduceMean should always return float32 tensor but TF doesn't
    // handled that conversion.
    if (x.dtype !== "float32") {
      x = this.cast(x, "float32");
    }
    // axesT is expected to be on CPU.
    const axesT = int32Small(axes);
    return execute0("Mean", [x, axesT], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
      ["Tidx", binding.ATTR_TYPE, binding.TF_INT32],
      ["keep_dims", binding.ATTR_BOOL, keepDims],
    ]);
  }

  reduceMax(x: TensorTF, axes: number[], keepDims: boolean): TensorTF
  {
    // axesT is expected to be on CPU.
    const axesT = int32Small(axes);
    return execute0("Max", [x, axesT], [
      ["T", binding.ATTR_TYPE, binding.getDType(x.handle)],
      ["Tidx", binding.ATTR_TYPE, binding.TF_INT32],
      ["keep_dims", binding.ATTR_BOOL, keepDims],
    ]);
  }

  equal(x: TensorTF, y: TensorTF): TensorTF {
    return execute1("Equal", [x, y]);
  }

  greater(x: TensorTF, y: TensorTF): TensorTF {
    return execute1("Greater", [x, y]);
  }

  greaterEqual(x: TensorTF, y: TensorTF): TensorTF {
    return execute1("GreaterEqual", [x, y]);
  }

  less(x: TensorTF, y: TensorTF): TensorTF {
    return execute1("Less", [x, y]);
  }

  lessEqual(x: TensorTF, y: TensorTF): TensorTF {
    return execute1("LessEqual", [x, y]);
  }

  select(cond: TensorTF, t: TensorTF, f: TensorTF): TensorTF {
    return execute1("Select", [cond, t, f], t.dtype);
  }

  sign(x: TensorTF): TensorTF {
    return execute1("Sign", [x]);
  }

  slice(x: TensorTF, begin: number[], size: number[]): TensorTF {
    let handle;
    // It seems that if x.dtype is int32 this must be done on CPU:
    // https://git.io/vNTSv
    if (x.dtype === "int32" &&
        !binding.getDevice(x.handle).endsWith("CPU:0")) {
      console.warn("Slice on GPU not supported for int32. Copying to CPU.");
      handle = binding.copyToDevice(ctx, x.handle, "CPU:0");
    } else {
      handle = x.handle;
    }

    // Slice op expects begin and size to reside on CPU.
    const beginT = int32Small(begin);
    const sizeT = int32Small(size);

    const handles = [handle, beginT.handle, sizeT.handle];
    const attrs = [
      ["T", binding.ATTR_TYPE, binding.getDType(handle)],
      ["Index", binding.ATTR_TYPE, binding.TF_INT32],
    ];
    const r = binding.execute(ctx, "Slice", attrs, handles);
    assertEqual(r.length, 1);
    return new TensorTF(r[0]);
  }

  gather(x: TensorTF, indices: TensorTF, axis: number): TensorTF {
    const axisT = int32Small(axis);
    const attrs = [
      ["Tparams", binding.ATTR_TYPE, dtypePropel2TF(x.dtype)],
      ["Tindices", binding.ATTR_TYPE, dtypePropel2TF(indices.dtype)],
      ["Taxis", binding.ATTR_TYPE, binding.TF_INT32],
    ];
    const handles = [x.handle, indices.handle, axisT.handle];
    const r = binding.execute(ctx, "GatherV2", attrs, handles);
    assertEqual(r.length, 1);
    return new TensorTF(r[0]);
  }

  concat(axis: number, inputs: TensorTF[]): TensorTF {
    const dtype = dtypePropel2TF(inputs[0].dtype);
    const handles = inputs.map(t => t.handle);
    handles.unshift(int32Small(axis).handle);
    const attrs = [
      ["T", binding.ATTR_TYPE, dtype],
      ["N", binding.ATTR_INT, inputs.length],
    ];
    const r = binding.execute(ctx, "Concat", attrs, handles);
    assertEqual(r.length, 1);
    return new TensorTF(r[0]);
  }

  reshape(x: TensorTF, newShape: types.Shape): TensorTF {
    // Reshape, like Slice, does not have an int32 GPU implementation
    // https://git.io/vNTd5
    let handle;
    if (x.dtype === "int32" &&
        !binding.getDevice(x.handle).endsWith("CPU:0")) {
      console.warn("Reshape on GPU not supported for int32. Copying to CPU.");
      handle = binding.copyToDevice(ctx, x.handle, "CPU:0");
    } else {
      handle = x.handle;
    }

    // Reshape op expects newShape to reside on CPU.
    const shapeT = int32Small(newShape);

    const handles = [handle, shapeT.handle];
    const attrs = [
      ["T", binding.ATTR_TYPE, binding.getDType(handle)],
      ["Tshape", binding.ATTR_TYPE, binding.TF_INT32],
    ];
    const r = binding.execute(ctx, "Reshape", attrs, handles);
    assertEqual(r.length, 1);
    return new TensorTF(r[0]);
  }

  softmax(x: TensorTF): TensorTF {
    return execute1("Softmax", [x]);
  }

  logSoftmax(x: TensorTF): TensorTF {
    return execute1("LogSoftmax", [x]);
  }

  cast(x: TensorTF, dtype: types.DType): TensorTF {
    return execute0("Cast", [x], [
      ["SrcT", binding.ATTR_TYPE, binding.getDType(x.handle)],
      ["DstT", binding.ATTR_TYPE, dtypePropel2TF(dtype)],
    ]);
  }

  oneHot(x: TensorTF, depth: number, onValue: number,
         offValue: number): TensorTF {
    if (x.dtype === "float32") {
      throw new Error("Must use integer type with oneHot.");
    }
    // OneHot expects depthT to be on CPU. However onT and offT need to be
    // colocated with x.
    const depthT = int32Small(depth);
    const onT = floatSmall(onValue, x);
    const offT = floatSmall(offValue, x);
    return execute0("OneHot", [x, depthT, onT, offT], [
      ["T", binding.ATTR_TYPE, binding.getDType(onT.handle)],
      ["TI", binding.ATTR_TYPE, binding.getDType(x.handle)],
      ["axis", binding.ATTR_INT, -1],
    ]);
  }

  conv2d(input: TensorTF, filter: TensorTF, opts: types.ConvOpts): TensorTF {
    return execute0("Conv2D", [input, filter], convAttrs(opts));
  }

  conv2dGradFilter(grad: TensorTF, input: TensorTF,
                   filterShape: types.Shape,
                   opts: types.ConvOpts): TensorTF {
    const filterShapeT = int32Small(filterShape);
    return execute0("Conv2DBackpropFilter",
                    [input, filterShapeT, grad],
                    convAttrs(opts));
  }

  conv2dGradInput(grad: TensorTF, inputShape: types.Shape,
                  filter: TensorTF, opts: types.ConvOpts): TensorTF {
    const inputShapeT = int32Small(inputShape);
    return execute0("Conv2DBackpropInput",
                    [inputShapeT, filter, grad],
                    convAttrs(opts));
  }

  maxPool(input: TensorTF, opts: types.PoolOpts): TensorTF {
    const attrs = poolAttrs(opts, binding.getDType(input.handle));
    return execute0("MaxPool", [input], attrs);
  }

  maxPoolGrad(grad: TensorTF, origInput: TensorTF, origOutput: TensorTF,
              opts: types.PoolOpts): TensorTF {
    const attrs = poolAttrs(opts, binding.getDType(origInput.handle));
    return execute0("MaxPoolGrad", [origInput, origOutput, grad], attrs);
  }
}

function poolAttrs(opts: types.PoolOpts, dtypeCode: DTypeCode): AttrDef[] {
  return [
    ["T", binding.ATTR_TYPE, dtypeCode],
    ["ksize", binding.ATTR_INT_LIST, tfStrides(opts.size)],
    ["strides", binding.ATTR_INT_LIST, tfStrides(opts.stride)],
    ["padding", binding.ATTR_STRING, opts.padding.toUpperCase()],
    ["data_format", binding.ATTR_STRING, "NHWC"],
  ];
}

function convAttrs(opts: types.ConvOpts): AttrDef[] {
  const dilations = [1, 1, 1, 1];  // TODO
  const padding = opts.padding.toUpperCase();
  return [
    ["T", binding.ATTR_TYPE, binding.TF_FLOAT],
    ["strides", binding.ATTR_INT_LIST, tfStrides(opts.stride)],
    ["use_cudnn_on_gpu", binding.ATTR_BOOL, false],
    ["padding", binding.ATTR_STRING, padding],
    ["data_format", binding.ATTR_STRING, "NHWC"],
    ["dilations", binding.ATTR_INT_LIST, dilations],
  ];
}

function tfStrides(s: number | [number, number]):
                   [number, number, number, number] {
  if (typeof s === "number") {
    return [1, s, s, 1];
  } else {
    assert(s.length === 2);
    return [1, s[0], s[1], 1];
  }
}
