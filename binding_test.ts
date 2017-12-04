import { assert, assertEqual, assertAllEqual } from "./util";
import * as fs from "fs";
import * as path from "path";

export function maybeRequireBinding() {
  // If we're in the browser, don't even attempt it.
  if (typeof window !== 'undefined') return null;

  // When using ts-node, we are in the root dir, after compiling to
  // javascript, we are in the dist dir.
  const toAttempt = [
    '../build/Debug/tensorflow-binding.node',
    '../build/Release/tensorflow-binding.node',
    './build/Debug/tensorflow-binding.node',
    './build/Release/tensorflow-binding.node',
  ];
  for (const fn of toAttempt) {
    if (fs.existsSync(path.join(__dirname, fn))) {
      return require(fn);
    }
  }
  return null;
}

const binding = maybeRequireBinding();
const ctx = new binding.Context();

function testEquals() {
  const a = new binding.Tensor(new Float32Array([2, 5]), [2]);
  const b = new binding.Tensor(new Float32Array([2, 4]), [2]);

  const opAttrs = [
    ["T", binding.ATTR_TYPE, binding.TF_FLOAT],
  ];
  const r = binding.execute(ctx, "Equal", opAttrs, [a, a])[0];
  assert(r.device == "CPU:0");
  assertAllEqual(r.shape, [2]);

  const result = Array.from(new Uint8Array(r.asArrayBuffer()));
  assertAllEqual(result, [1, 1]);

  const r2 = binding.execute(ctx, "Equal", opAttrs, [a, b])[0];
  const result2 = Array.from(new Uint8Array(r2.asArrayBuffer()));
  assertAllEqual(result2, [1, 0]);
}

function testMatMul() {
  assert(ctx instanceof binding.Context);

  const typedArray = new Float32Array([1, 2, 3, 4, 5, 6]);
  const a = new binding.Tensor(typedArray, [2, 3]);
  const b = new binding.Tensor(typedArray, [3, 2]);
  assert(a.device == "CPU:0");
  assert(b.device == "CPU:0");
  assertAllEqual(a.shape, [2, 3]);
  assertAllEqual(b.shape, [3, 2]);

  const opAttrs = [
    ["transpose_a", binding.ATTR_BOOL, false],
    ["transpose_b", binding.ATTR_BOOL, false],
    ["T", binding.ATTR_TYPE, binding.TF_FLOAT],
  ];
  const retvals = binding.execute(ctx, "MatMul", opAttrs, [a, b]);
  const r = retvals[0];
  assert(r.device == "CPU:0");
  const result = Array.from(new Float32Array(r.asArrayBuffer()));
  assertAllEqual(result, [22, 28, 49, 64]);
}

function testMul() {
  const typedArray = new Float32Array([2, 5]);
  const a = new binding.Tensor(typedArray, [2]);
  const b = new binding.Tensor(typedArray, [2]);
  assert(a.device == "CPU:0");
  assert(b.device == "CPU:0");

  assert(a.dtype == binding.TF_FLOAT);
  assert(b.dtype == binding.TF_FLOAT);

  const opAttrs = [
    ["T", binding.ATTR_TYPE, binding.TF_FLOAT],
  ];
  const retvals = binding.execute(ctx, "Mul", opAttrs, [a, b]);
  const r = retvals[0];
  assert(r.device == "CPU:0");
  const result = Array.from(new Float32Array(r.asArrayBuffer()));
  assertAllEqual(result, [4, 25]);
}

function testChaining() {
  // Do an Equal followed by ReduceAll.
  const a = new binding.Tensor(new Float32Array([2, 5]), [2]);
  const b = new binding.Tensor(new Float32Array([2, 4]), [2]);

  const opAttrs = [
    ["T", binding.ATTR_TYPE, binding.TF_FLOAT],
  ];
  const r = binding.execute(ctx, "Equal", opAttrs, [a, a])[0];
  assert(r.dtype == binding.TF_BOOL);
  assertAllEqual(r.shape, [2]);

  const reductionIndices = new binding.Tensor(new Int32Array([0]), [1]);

  const opAttrs2 = [
    ["Tidx", binding.ATTR_TYPE, binding.TF_INT32],
    ["keep_dims", binding.ATTR_BOOL, false],
  ];
  const r2 = binding.execute(ctx, "All", opAttrs2, [r, reductionIndices])[0];
  const result2 = Array.from(new Uint8Array(r2.asArrayBuffer()));
  assertAllEqual(result2, [1]);
}

testEquals();
testMatMul();
testMul();
testChaining();
