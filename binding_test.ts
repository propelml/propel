import { assert, assertAllEqual } from "./util";
import * as fs from "fs";
import * as path from "path";

function requireBinding() {
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
  assert(false, "Count not find tensorflow-binding.node.");
}

const binding = requireBinding();

function testMatMul() {
  const ctx = new binding.Context();
  assert(ctx instanceof binding.Context);

  const typedArray = new Float32Array([1, 2, 3, 4, 5, 6]);
  const a = new binding.Tensor(typedArray, [2, 3]);
  const b = new binding.Tensor(typedArray, [3, 2]);
  assert(a.device == "CPU:0");
  assert(b.device == "CPU:0");

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
  console.log(result);
}

testMatMul();
