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
import { test } from "../tools/tester";
import * as tf from "./tf";
import { assert, assertAllEqual } from "./util";

assert(tf.loadBinding());
const binding = tf.binding;
const ctx = tf.ctx;

test(async function binding_version() {
  assert(binding.tensorflowVersion);
  console.log(`Tensorflow version: ${binding.tensorflowVersion}`);
});

test(async function binding_equals() {
  const a = new binding.Handle(new Float32Array([2, 5]), [2], binding.TF_FLOAT);
  const b = new binding.Handle(new Float32Array([2, 4]), [2], binding.TF_FLOAT);

  const opAttrs = [
    ["T", binding.ATTR_TYPE, binding.TF_FLOAT],
  ];
  const r = binding.execute(ctx, "Equal", opAttrs, [a, a])[0];
  assert(binding.getDevice(r) === "CPU:0");
  assertAllEqual(binding.getShape(r), [2]);

  const result = Array.from(new Uint8Array(binding.asArrayBuffer(r)));
  assertAllEqual(result, [1, 1]);

  const r2 = binding.execute(ctx, "Equal", opAttrs, [a, b])[0];
  const result2 = Array.from(new Uint8Array(binding.asArrayBuffer(r2)));
  assertAllEqual(result2, [1, 0]);
});

test(async function binding_matMul() {
  assert(ctx instanceof binding.Context);

  const typedArray = new Float32Array([1, 2, 3, 4, 5, 6]);
  const a = new binding.Handle(typedArray, [2, 3], binding.TF_FLOAT);
  const b = new binding.Handle(typedArray, [3, 2], binding.TF_FLOAT);
  assert(binding.getDevice(a) === "CPU:0");
  assert(binding.getDevice(b) === "CPU:0");
  assertAllEqual(binding.getShape(a), [2, 3]);
  assertAllEqual(binding.getShape(b), [3, 2]);

  const opAttrs = [
    ["transpose_a", binding.ATTR_BOOL, false],
    ["transpose_b", binding.ATTR_BOOL, false],
    ["T", binding.ATTR_TYPE, binding.TF_FLOAT],
  ];
  const retvals = binding.execute(ctx, "MatMul", opAttrs, [a, b]);
  const r = retvals[0];
  assert(binding.getDevice(r) === "CPU:0");
  const result = Array.from(new Float32Array(binding.asArrayBuffer(r)));
  assertAllEqual(result, [22, 28, 49, 64]);
});

test(async function binding_mul() {
  const typedArray = new Float32Array([2, 5]);
  const a = new binding.Handle(typedArray, [2], binding.TF_FLOAT);
  const b = new binding.Handle(typedArray, [2], binding.TF_FLOAT);
  assert(binding.getDevice(a) === "CPU:0");
  assert(binding.getDevice(b) === "CPU:0");

  assert(binding.getDType(a) === binding.TF_FLOAT);
  assert(binding.getDType(b) === binding.TF_FLOAT);

  const opAttrs = [
    ["T", binding.ATTR_TYPE, binding.TF_FLOAT],
  ];
  const retvals = binding.execute(ctx, "Mul", opAttrs, [a, b]);
  const r = retvals[0];
  assert(binding.getDevice(r) === "CPU:0");
  const result = Array.from(new Float32Array(binding.asArrayBuffer(r)));
  assertAllEqual(result, [4, 25]);
});

test(async function binding_chaining() {
  // Do an Equal followed by ReduceAll.
  const a = new binding.Handle(new Float32Array([2, 5]), [2], binding.TF_FLOAT);
  const b = new binding.Handle(new Float32Array([2, 5]), [2], binding.TF_FLOAT);

  const opAttrs = [
    ["T", binding.ATTR_TYPE, binding.TF_FLOAT],
  ];
  const r = binding.execute(ctx, "Equal", opAttrs, [a, b])[0];
  assert(binding.getDType(r) === binding.TF_BOOL);
  assertAllEqual(binding.getShape(r), [2]);

  const reductionIndices = new binding.Handle(new Int32Array([0]), [1],
                                              binding.TF_INT32);
  const opAttrs2 = [
    ["Tidx", binding.ATTR_TYPE, binding.TF_INT32],
    ["keep_dims", binding.ATTR_BOOL, false],
  ];
  const r2 = binding.execute(ctx, "All", opAttrs2, [r, reductionIndices])[0];
  const result2 = Array.from(new Uint8Array(binding.asArrayBuffer(r2)));
  assertAllEqual(result2, [1]);
});

test(async function binding_reshape() {
  const typedArray = new Float32Array([1, 2, 3, 4, 5, 6]);
  const t = new binding.Handle(typedArray, [2, 3], binding.TF_FLOAT);
  const shape = new binding.Handle(new Int32Array([3, 2]), [2],
                                   binding.TF_INT32);
  const opAttrs = [
    ["T", binding.ATTR_TYPE, binding.TF_FLOAT],
    ["Tshape", binding.ATTR_TYPE, binding.TF_INT32],
  ];
  const r = binding.execute(ctx, "Reshape", opAttrs, [t, shape])[0];
  assertAllEqual(binding.getShape(r), [3, 2]);
});

test(async function binding_boolean() {
  const ta = new Uint8Array([0, 1, 0, 1]);
  const t = new binding.Handle(ta, [3], binding.TF_BOOL);
  assert(binding.getDType(t) === binding.TF_BOOL);
  assertAllEqual(binding.getShape(t), [3]);
  const result = Array.from(new Uint8Array(binding.asArrayBuffer(t)));
  assertAllEqual(result, [0, 1, 0, 1]);
});

test(async function binding_listDevices() {
  const devices = binding.listDevices(ctx);
  assert(devices.length >= 1);
  // Assuming first device is always CPU.
  const cpuDevice = devices[0];
  assert(cpuDevice["deviceType"] === "CPU");
  assert(cpuDevice["name"].indexOf("device:CPU:0") > 0);
  assert(cpuDevice["memoryBytes"] > 1024);
  console.log(devices);
});

test(async function binding_copyToDevice() {
  // Only do this test if there's more than one device.
  const devices = binding.listDevices(ctx);
  if (devices.length < 2) {
    console.log("Skipping testCopyToDevice because there's only one device.");
    return;
  }

  const t = new binding.Handle(new Float32Array([1, 2]), [2], binding.TF_FLOAT);

  const tGPU = binding.copyToDevice(ctx, t, "GPU:0");
  assertAllEqual(binding.getShape(tGPU), [2]);
  console.log("tGPU device", binding.getDevice(tGPU));
  assert(binding.getDevice(tGPU).endsWith("GPU:0"));
  console.log("copyToDevice ok");
});

test(async function binding_createSmallHandle() {
  const types: Array<[number, any]> = [
    [binding.TF_FLOAT, Float32Array],
    [binding.TF_INT32, Int32Array]
  ];

  let h, v, hCpu;
  for (const [tftype, taConstructor] of types) {
    // scalar CPU
    h = binding.createSmallHandle(ctx, tftype, "CPU:0", 42);
    assert(binding.getDevice(h) === "CPU:0");
    assert(binding.getShape(h).length === 0);
    assert(binding.getDType(h) === tftype);
    v = Array.from(new taConstructor(binding.asArrayBuffer(h)));
    assertAllEqual(v, [42]);

    // array CPU
    h = binding.createSmallHandle(ctx, tftype, "CPU:0", [1, 2, 3]);
    assert(binding.getDevice(h) === "CPU:0");
    assertAllEqual(binding.getShape(h), [3]);
    assert(binding.getDType(h) === tftype);
    v = Array.from(new taConstructor(binding.asArrayBuffer(h)));
    assertAllEqual(v, [1, 2, 3]);

    // Figure out if we have a GPU to test.
    const devices = binding.listDevices(ctx);
    if (devices.length < 2) {
      console.log("Skip rest of testCreateSmallHandle, no GPU for testing.");
      return;
    }

    // scalar GPU
    h = binding.createSmallHandle(ctx, tftype, "GPU:0", 42);
    assert(binding.getDevice(h).endsWith("GPU:0"));
    assert(binding.getShape(h).length === 0);
    assert(binding.getDType(h) === tftype);
    hCpu = binding.copyToDevice(ctx, h, "CPU:0");
    v = Array.from(new taConstructor(binding.asArrayBuffer(hCpu)));
    assertAllEqual(v, [42]);

    // array GPU
    h = binding.createSmallHandle(ctx, tftype, "GPU:0", [1, 2, 3]);
    assert(binding.getDevice(h).endsWith("GPU:0"));
    assertAllEqual(binding.getShape(h), [3]);
    assert(binding.getDType(h) === tftype);
    hCpu = binding.copyToDevice(ctx, h, "CPU:0");
    v = Array.from(new taConstructor(binding.asArrayBuffer(hCpu)));
    assertAllEqual(v, [1, 2, 3]);

    console.log("testCreateSmallHandle", taConstructor.name, "ok");
  }
});

test(async function testDispose() {
  const a = new binding.Handle(new Float32Array([2, 5]), [2], binding.TF_FLOAT);
  binding.dispose(a);

  let didThrow = false;
  try {
  binding.dispose(null);
  } catch (e) {
    didThrow = true;
  }
  assert(didThrow);
});
