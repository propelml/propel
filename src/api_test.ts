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
import { fill, grad, linspace, listDevices, multigrad, ones,
  Params, randn, range, tensor, Tensor, TensorLike, zeros } from "./api";
import * as api from "./api";
import { assertAllClose, assertAllEqual, assertClose,
  assertShapesEqual } from "./tensor_util";
import * as types from "./types";
import { assert, IS_NODE } from "./util";

function checkGrad(f, g, val = 1.0) {
  const epsilon = 0.01;
  const a = tensor(f(val + epsilon));
  const b = tensor(f(val - epsilon));
  const expected = a.sub(b).div(2 * epsilon);
  const actual = g(val);
  assertClose(actual, expected);
}

function gpuAvail(): boolean {
  return listDevices().length === 1 ? false : true;
}

// This is the type signature of the $ and gpuConvert.
interface ConvertFn {
  (x: TensorLike, args?: types.TensorOpts): Tensor;
}

function gpuConvert(x: TensorLike, args?: types.TensorOpts): Tensor {
  return tensor(x, args).gpu();
}

// Allows tests to run on CPU:0 and GPU:0 (if available).
function testDevices(
  fn: ($: ConvertFn, device: string) => Promise<void>
): void {
  test({ fn: () => fn(tensor, "CPU:0"), name: `${fn.name}_cpu` });
  if (gpuAvail()) {
    test({ fn: () => fn(gpuConvert, "GPU:0"), name: `${fn.name}_gpu` });
  }
}

// Basic Tests

test(async function api_linspace() {
  const x = linspace(-4, 4, 6);
  assertAllClose(x, [-4., -2.4, -0.8,  0.8,  2.4, 4.]);
});

test(async function api_range() {
  const r1 = range(-2, 2);
  assertAllEqual(r1, [-2, -1, 0, 1]);
  const r2 = range(4);
  assertAllEqual(r2, [0, 1, 2, 3]);
  const r3 = range(4, 10, 2);
  assertAllEqual(r3, [4, 6, 8]);
});

test(async function api_randn() {
  const t = randn([2, 3]);
  assertAllEqual(t.shape, [2, 3]);
  const d = t.dataSync();
  console.log("randn", d);
  // TODO this isn't the best test...
  assert(d[0] !== d[1]);
  assert(d[1] !== d[2]);
});

test(async function api_convertWithType() {
  const t = tensor([1, 2, 3], {dtype: "int32"});
  assert(t.dtype === "int32");
  const ta = t.dataSync();
  assert(ta instanceof Int32Array);

  const ta2 = new Int32Array([1, 2, 3]);
  const t2 = tensor(ta2);
  assert(t2.dtype === "int32");
  assert(t2.dataSync() instanceof Int32Array);
});

// Backprop Tests

test(async function api_inc() {
  function f(x) {
    return tensor(x).add(1);
  }
  assertClose(f(1), 2);
  assertClose(f(-1), 0);
  const g = grad(f);
  assertClose(g(1.0), 1.);
  checkGrad(f, g, 1.0);
});

test(async function api_mul() {
  const f = (x) => tensor(42).mul(x);
  assertClose(f(1), 42);
  assertClose(f(2), 84);
  const g = grad(f);
  assertClose(g(1.), 42.);
  checkGrad(f, g, 1.0);
});

test(async function api_squared() {
  // f(x) = x^2
  function f(x) {
    return tensor(x).mul(x);
  }
  assertClose(f(1), 1);
  assertClose(f(16), 256);
  const g = grad(f); // g(x) = f'(x) = 2x
  assertClose(g(1), 2);
  assertClose(g(10), 20);
  checkGrad(f, g, 1.0);
});

test(async function api_squaredMatrix() {
  // f(x) = x^2
  function f(x) {
    return tensor(x).mul(x);
  }
  assertAllEqual(f([[1, 2], [3, 4]]), [[1, 4], [9, 16]]);
  const g = grad(f); // g(x) = f'(x) = 2x
  const v = g([[1, 2], [3, 4]]);
  assertAllEqual(v.shape, [2, 2]);
  assertAllEqual(v, [[2, 4], [6, 8]]);
});

test(async function api_div() {
  // f(x) = (1 + x) / x
  function f(x) {
    x = tensor(x);
    return x.add(1).div(x);
  }
  assertClose(f(1), 2);
  assertClose(f(16), (1 + 16) / 16);
  const g = grad(f); // g(x) = -1 / x^2
  assertClose(g(1), -1);
  assertClose(g(10), -1 / 100);
  checkGrad(f, g, 1.0);
});

test(async function api_constant() {
  const f = (_) => 42;
  assertClose(f(1), 42);
  assertClose(f(-1), 42);
  const g = grad(f);
  assertClose(g(1.0), 0.);
  checkGrad(f, g, 1.0);
});

test(async function api_exp() {
  // f(x) = exp(1+x)
  function f(x) {
    return tensor(x).add(1).exp();
  }
  assertClose(f(1), 7.3890);
  assertClose(f(2), 20.0855);
  const g = grad(f); // g == f
  assertClose(g(1), 7.3890);
  assertClose(g(2), 20.0855);
  checkGrad(f, g, 1.0);
});

test(async function api_log() {
  // f(x) = log(x)/log(base)
  function f(x, base) {
    return tensor(x).log().div(tensor(base).log());
  }
  assertClose(f(2, 2), 1);
  assertClose(f(9, 3), 2);
  assertClose(f(64, 4), 3);
  assertClose(f(625, 5), 4);
  const g = grad(f); // g = (x*Math.log(base))^-1
  assertClose(g(2, 2), 1 / (2 * Math.log(2)));
  assertClose(g(9, 3), 1 / (9 * Math.log(3)));
  assertClose(g(64, 4), 1 / (64 * Math.log(4)));
  assertClose(g(625, 5), 1 / (625 * Math.log(5)));
});

test(async function api_sub() {
  function f(x) {
    return tensor(1).sub(x);
  }
  assertClose(f(1), 0);
  assertClose(f(2), -1);
  const g = grad(f);
  assertClose(g(1), -1);
  assertClose(g(2), -1);
  checkGrad(f, g, 1.0);
});

test(async function api_div2() {
  function f(x) {
    x = tensor(x);
    return tensor(1).sub(x).div(x.add(1));
  }
  assertClose(f(1), 0);
  assertClose(f(2), -1 / 3);
  const g = grad(f); // g(x) = -2 / (x + 1)^2
  assertClose(g(1), -2 / 4);
  assertClose(g(2), -2 / 9);
  checkGrad(f, g, 1.0);
});

test(async function api_div3() {
  function f(x) {
    const y = tensor(x).exp();
    return y.div(y);
  }
  assertClose(f(1), 1.);
  assertClose(f(2), 1.);
  const g = grad(f);
  assertClose(g(1), 0.);
  assertClose(g(2), 0.);
  checkGrad(f, g, 1.0);
});

test(async function api_tanh() {
  const f = (x) => tensor(x).tanh();
  assertClose(f(1), 0.7615);
  assertClose(f(16), 0.9999);
  const g = grad(f);
  assertClose(g(1), 0.4199);
  checkGrad(f, g, 1.0);
});

test(async function api_relu() {
  const f = (x) => tensor(x).relu();
  assertAllEqual(f([-5, 0, 5]), [0, 0, 5]);
  const g = grad(f);
  assertAllClose(g([-5, 0.1, 5]), [0, 1, 1]);
  checkGrad(f, g, 1.0);
  checkGrad(f, g, -1.0);
  const g2 = grad(g);
  assertAllClose(g2([-5, 0.1, 5]), [0, 0, 0]);
});

test(async function api_sigmoid() {
  const f = (x) => tensor(x).sigmoid();
  assertAllClose(f([-1, 0, 1]), [0.26894142, 0.5, 0.73105858]);
  const g = grad(f);
  assertAllClose(g([-1, 0, 1]), [0.19661193, 0.25, 0.19661193]);
  checkGrad(f, g, 1.0);
  checkGrad(f, g, -1.0);
});

test(async function api_abs() {
  const f = (x) => tensor(x).abs();
  assertAllEqual(f([-5, 0, 5]), [5, 0, 5]);
  const g = grad(f);
  assertAllClose(g([-5, 0.1, 5]), [-1, 1, 1]);
  checkGrad(f, g, 1.0);
  checkGrad(f, g, -1.0);
});

test(async function api_multigrad() {
  function f(a, b) {
    return tensor(a).mul(2).add(tensor(b).mul(3));
  }
  assertClose(f(1, 1), 5);
  assertClose(f(1, 2), 8);
  const g = multigrad(f);
  assertClose(g(1, 1)[0], 2);
  assertClose(g(1, 1)[1], 3);
  assertClose(g(4, 2)[0], 2);
  assertClose(g(4, 2)[1], 3);
});

test(async function api_gradGradTanh() {
  const f = (x) => tensor(x).tanh();
  assertAllClose(f([1, 16]), [0.7615, 0.9999]);
  const g = grad(grad(f));
  // def g(x): return -2 * np.tanh(x) / np.square(np.cosh(x))
  assertAllClose(g([1, 2]), [-0.6397, -0.13621]);
});

test(async function api_sinh() {
  const f = (x) => tensor(x).sinh();
  const v = tensor([1, 2]);
  assertAllClose(f(v), [1.17520119,  3.62686041]);
  // The derivtive of sinh is cosh.
  const g = grad(f);
  assertAllClose(g(v), v.cosh());
});

test(async function api_fill() {
  const f = (x) => fill(x, [2, 3]);
  assertAllEqual(f(1), [[1, 1, 1], [1, 1, 1]]);
  assertAllEqual(f(42), [[42, 42, 42], [42, 42, 42]]);
  // TODO
  // const g = grad(f);
  // assertAllEqual(g(1), [1]);
});

testDevices(async function api_square_sqrt_pow(tensor, device) {
  const square = x => x.square();
  const sqrt = x => x.sqrt();
  const pow3 = x => x.pow(3);
  const v = tensor([4, 16]);
  assertAllClose(square(v), [16, 256]);
  assertAllClose(sqrt(v), [2, 4]);
  assertAllClose(pow3(v), [64, 4096]);
  assertAllClose(grad(square)(v), [8, 32]);
  assertAllClose(grad(sqrt)(v), [1 / 4, 1 / 8]);
  assertAllClose(grad(pow3)(v), [48, 768]);
  // TODO There's a bug in backprop when taking the gradient wrt to the
  // exponent on WebGL.
  // const exp3 = e => tensor(3).pow(e);
  // assertAllClose(grad(exp3)([0, 1]), [ 1.09861229,  3.29583687]);
});

test(async function api_transpose() {
  const f = (x) => tensor(x).transpose();
  const a = tensor([[1, 2], [3, 4]]);
  const aT = tensor([[1, 3], [2, 4]]);
  assertAllEqual(f(a), aT);
  const g = grad(f);
  assertAllEqual(g(a), [[1, 1], [1, 1]]);

  const f2 = (x) => tensor(x).transpose().mul(2);
  const g2 = grad(f2);
  assertAllEqual(g2(a), [[2, 2], [2, 2]]);
});

test(async function api_reverse() {
  assertAllEqual(tensor([1, 2, 3, 4]).reverse(), [4, 3, 2, 1]);

  const t = tensor([[
    [[ 0,  1,  2,  3],
     [ 4,  5,  6,  7],
     [ 8,  9, 10, 11]],
    [[12, 13, 14, 15],
     [16, 17, 18, 19],
     [20, 21, 22, 23]]
  ]]);
  assertAllEqual(t.shape, [1, 2, 3, 4]);
  const tR1 = tensor([[
    [[12, 13, 14, 15],
     [16, 17, 18, 19],
     [20, 21, 22, 23]],
    [[ 0,  1,  2,  3],
     [ 4,  5,  6,  7],
     [ 8,  9, 10, 11]]
  ]]);
  assertAllEqual(t.reverse([1]), tR1);
  assertAllEqual(t.reverse([-3]), tR1);
  const tR2 = tensor([[
    [[ 8,  9, 10, 11],
     [ 4,  5,  6,  7],
     [ 0,  1,  2,  3]],
    [[20, 21, 22, 23],
     [16, 17, 18, 19],
     [12, 13, 14, 15]]
  ]]);
  assertAllEqual(t.reverse([2]), tR2);
  assertAllEqual(t.reverse([-2]), tR2);
  const tR3 = tensor([[
    [[3,  2,  1,  0],
     [7,  6,  5,  4],
     [11, 10, 9, 8]],
    [[15, 14, 13, 12],
     [19, 18, 17, 16],
     [23, 22, 21, 20]]
  ]]);
  assertAllEqual(t.reverse([3]), tR3);
  assertAllEqual(t.reverse([-1]), tR3);

  const f = (x) => tensor(x).reverse().mul(2);
  const g = grad(f);
  assertAllEqual(g([1, 2, 3]), [2, 2, 2]);
});

test(async function api_matMul() {
  function f(x, y) {
    return tensor(x).matmul(y);
  }
  const a = tensor([
    [9, 8, 7],
    [6, 5, 4],
  ]);
  const b = tensor([
    [1, 2],
    [4, 5],
    [7, 8],
  ]);
  const r = f(a, b);
  assertShapesEqual(r.shape, [2, 2]);
  assertAllClose(r, [
    [90, 114],
    [54, 69],
  ]);
  // Now test gradients
  const g = multigrad(f, [0, 1]);
  const gab = g(a, b);
  assertAllEqual(gab[0], [
    [3, 9, 15],
    [3, 9, 15],
  ]);
  assertAllEqual(gab[1], [
    [15, 15],
    [13, 13],
    [11, 11],
  ]);
});

testDevices(async function api_reduceSum(tensor, device) {
  const a = tensor([
    [9, 8, 7],
    [6, 5, 4],
  ]);
  assertAllEqual(a.reduceSum([0]), [9 + 6, 8 + 5, 7 + 4]);
  assertAllEqual(a.reduceSum([1]), [9 + 8 + 7, 6 + 5 + 4]);
  assertAllEqual(a.reduceSum(), 9 + 8 + 7 + 6 + 5 + 4);

  assertAllEqual(a.reduceSum([0], true), [[9 + 6, 8 + 5, 7 + 4]]);
  assertAllEqual(a.reduceSum([1], true), [[9 + 8 + 7], [6 + 5 + 4]]);

  const f = (x) => tensor(x).mul(2).reduceSum([0]);
  const g = grad(f);
  assertAllEqual(g(a), [[2, 2, 2], [2, 2, 2]]);

  const b = tensor([
    [9, 8, 7],
    [6, 5, 4],
    [1, 2, 3],
    [4, -4, -5],
  ]);
  const f2 = (x) => tensor(x).reduceSum([1]);
  assertShapesEqual(f2(b).shape, [4]);
  const g2 = grad(f2);
  assertAllEqual(g2(b), [
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
  ]);
});

testDevices(async function api_reduceMean(tensor, device) {
  const a = tensor([
    [9, 8, 7],
    [6, 5, 4],
  ]);
  assert(a.device === device);
  assertAllEqual(a.reduceMean([0]), [7.5, 6.5, 5.5]);
  assertAllEqual(a.reduceMean([1]), [8, 5]);
  assertAllEqual(a.reduceMean(), 6.5);

  assertAllEqual(a.reduceMean([0], true), [[7.5, 6.5, 5.5]]);
  assertAllEqual(a.reduceMean([1], true), [[8], [5]]);

  const f = (x) => tensor(x).mul(2).reduceMean([0]);
  const g = grad(f);
  assertAllEqual(g(a), [[1, 1, 1], [1, 1, 1]]);

  const b = tensor([
    [9, 8, 7],
    [6, 5, 4],
    [1, 2, 3],
    [4, -4, -5],
  ]);
  const f2 = (x) => tensor(x).reduceMean([1]);
  assertShapesEqual(f2(b).shape, [4]);
  const g2 = grad(f2);
  const t = 1 / 3;
  assertAllClose(g2(b), [
    [t, t, t],
    [t, t, t],
    [t, t, t],
    [t, t, t],
  ]);
  assertAllEqual(api.uint8([255, 255, 255]).reduceMean(), 255);
});

testDevices(async function api_reduceMax(tensor, device) {
  const a = tensor([
    [9, 5, 7],
    [6, 8, 4],
  ]);
  assertAllEqual(a.reduceMax([0]), [9, 8, 7]);
  assertAllEqual(a.reduceMax([1]), [9, 8]);
  assertAllEqual(a.reduceMax(), 9);
  assertAllEqual(a.reduceMax([0], true), [[9, 8, 7]]);
  assertAllEqual(a.reduceMax([1], true), [[9], [8]]);

  /* TODO
  const f = (x) => tensor(x).reduceMax([0])
  const g = grad(f);
  assertAllEqual(g(a), [[1, 0, 1], [0, 1, 0]]);
  */
});

testDevices(async function api_onesAndZerosLike(tensor, device) {
  const a = tensor([
    [9, 5, 7],
    [6, 8, 4],
  ]);
  const ones = a.onesLike();
  const zeros = a.zerosLike();
  assert(ones.device === device);
  assert(zeros.device === device);
  assertAllEqual(ones, [ [1, 1, 1], [1, 1, 1] ]);
  assertAllEqual(zeros, [ [0, 0, 0], [0, 0, 0] ]);
});

test(async function api_equal() {
  const a = tensor([
    [9, 5, 7],
    [6, 8, 4],
  ]);
  const b = tensor([
    [9, 3, 7],
    [0, 8, 2],
  ]);
  const r = a.equal(b);
  assert(r.dtype === "bool");
  // TODO Allow assertAllEqual to handle boolean.
  assertAllEqual(r, [ [1, 0, 1], [0, 1, 0] ]);

  // equal isn't differentiable but it should have the same behavior as
  // autograd does.
  const f = (x, y) => tensor(x).equal(y);
  const g = multigrad(f, [0, 1]);
  assertAllEqual(g(a, b)[0], [ [0, 0, 0], [0, 0, 0] ]);
  assertAllEqual(g(a, b)[1], [ [0, 0, 0], [0, 0, 0] ]);
});

test(async function api_greater() {
  const a = tensor([
    [9, 5, 7],
    [6, 8, 2],
  ]);
  const b = tensor([
    [9, 3, 7],
    [0, 8, 4],
  ]);
  const r = a.greater(b);
  assert(r.dtype === "bool");
  // TODO Allow assertAllEqual to handle boolean.
  assertAllEqual(r, [ [0, 1, 0], [1, 0, 0] ]);
  // greater isn't differentiable but it should have the same behavior as
  // autograd does.
  const f = (x, y) => tensor(x).greater(y);
  const g = multigrad(f, [0, 1]);
  assertAllEqual(g(a, b)[0], [ [0, 0, 0], [0, 0, 0] ]);
  assertAllEqual(g(a, b)[1], [ [0, 0, 0], [0, 0, 0] ]);
});

test(async function api_greaterEqual() {
  const a = tensor([
    [9, 5, 7],
    [6, 8, 2],
  ]);
  const b = tensor([
    [9, 3, 7],
    [0, 8, 4],
  ]);
  const r = a.greaterEqual(b);
  assert(r.dtype === "bool");
  // TODO Allow assertAllEqual to handle boolean.
  assertAllEqual(r, [ [1, 1, 1], [1, 1, 0] ]);
  // greaterEqual isn't differentiable but it should have the same behavior as
  // autograd does.
  const f = (x, y) => tensor(x).greaterEqual(y);
  const g = multigrad(f, [0, 1]);
  assertAllEqual(g(a, b)[0], [ [0, 0, 0], [0, 0, 0] ]);
  assertAllEqual(g(a, b)[1], [ [0, 0, 0], [0, 0, 0] ]);
});

test(async function api_less() {
  const a = tensor([
    [9, 5, 7],
    [6, 8, 2],
  ]);
  const b = tensor([
    [9, 3, 7],
    [0, 8, 4],
  ]);
  const r = a.less(b);
  assert(r.dtype === "bool");
  // TODO Allow assertAllEqual to handle boolean.
  assertAllEqual(r, [ [0, 0, 0], [0, 0, 1] ]);
  // less isn't differentiable but it should have the same behavior as
  // autograd does.
  const f = (x, y) => tensor(x).less(y);
  const g = multigrad(f, [0, 1]);
  assertAllEqual(g(a, b)[0], [ [0, 0, 0], [0, 0, 0] ]);
  assertAllEqual(g(a, b)[1], [ [0, 0, 0], [0, 0, 0] ]);
});

test(async function api_lessEqual() {
  const a = tensor([
    [9, 5, 7],
    [6, 8, 2],
  ]);
  const b = tensor([
    [9, 3, 7],
    [0, 8, 4],
  ]);
  const r = a.lessEqual(b);
  assert(r.dtype === "bool");
  // TODO Allow assertAllEqual to handle boolean.
  assertAllEqual(r, [ [1, 0, 1], [0, 1, 1] ]);
  // lessEqual isn't differentiable but it should have the same behavior as
  // autograd does.
  const f = (x, y) => tensor(x).lessEqual(y);
  const g = multigrad(f, [0, 1]);
  assertAllEqual(g(a, b)[0], [ [0, 0, 0], [0, 0, 0] ]);
  assertAllEqual(g(a, b)[1], [ [0, 0, 0], [0, 0, 0] ]);
});

test(async function api_select() {
  const t = tensor([
    [1, 2, 3],
    [4, 5, 6],
  ]);
  const f = tensor([
    [ 7,  8,  9],
    [10, 11, 12],
  ]);
  // TODO Use false/true literals instead of 0 and 1 in cond.
  const cond = tensor([
    [1, 0, 1],
    [0, 1, 0],
  ], {dtype: "bool"});
  const r = cond.select(t, f);
  assertAllEqual(r, [
    [ 1, 8,  3],
    [10, 5, 12],
  ]);
  // select isn't differentiable.
  const g = grad((c) => c.select(t, f));
  assertAllEqual(g(cond), [ [0, 0, 0], [0, 0, 0] ]);

  function f2(x) {
    x = tensor(x);
    const y = x.sub(1).relu();
    const z = tensor(-1).sub(x).relu();
    return x.greater(x.zerosLike()).select(y, z);
  }
  assertAllEqual(grad(f2)([-3, -0.5, 0.5, 3]),
                 [-1, 0, 0, 1]);
});

test(async function api_sign() {
  const x = tensor([-2, 5, -1, 3]);
  assertAllEqual(x.sign(), [-1, 1, -1, 1]);
  // sign isn't differentiable.
  const g = grad((c) => c.sign());
  assertAllEqual(g(x), [0, 0, 0, 0]);
});

testDevices(async function api_reshape(tensor, device) {
  const a = tensor([
    [9, 5, 7],
    [6, 8, 4],
  ]);
  assertAllEqual(a.reshape([3, 2]), [
    [9, 5],
    [7, 6],
    [8, 4],
  ]);
  const f = (x) => tensor(x).reshape([3, 2]);
  const g = grad(f);
  const ga = g(a);
  assert(ga.device === device);
  assertAllEqual(ga, [
    [1, 1, 1],
    [1, 1, 1],
  ]);
});

testDevices(async function api_expandDims(tensor, device) {
  const a = tensor([[1, 2, 3], [4, 5, 6]]);
  assertAllEqual(a.expandDims(0), [[[1, 2, 3], [4, 5, 6]]]);
  assertAllEqual(a.expandDims(1), [[[1, 2, 3]], [[4, 5, 6]]]);
  assertAllEqual(a.expandDims(2), [[[1], [2], [3]], [[4], [5], [6]]]);
  assertAllEqual(a.expandDims(-1), [[[1], [2], [3]], [[4], [5], [6]]]);
});

test(async function api_flatten() {
  const a = tensor([[1, 2], [3, 4]]);
  assertAllEqual(a.flatten(), [1, 2, 3, 4]);
});

test(async function api_squeeze() {
  const a = tensor([[[0], [1], [2]]]);
  assertShapesEqual(a.shape, [1, 3, 1]);
  const b = a.squeeze();
  assertAllEqual(b, [0, 1, 2]);
  const c = tensor([[1, 2], [3, 4]]);
  assertAllEqual(c.squeeze(), c);
});

test(async function api_reduceLogSumExp() {
  assertClose(tensor([1, 2, 3, 4]).reduceLogSumExp(), 4.44018969856);
  const f = (x) => tensor(x).reduceLogSumExp();
  const g = grad(f);
  assertAllClose(g([2, 3]), [0.26894142, 0.73105858]);
});

test(async function api_softmax() {
  const f = (x) => tensor(x).softmax();
  assertAllClose(f([1, 2, 3, 4]),
    [0.0320586, 0.08714432, 0.23688281, 0.64391422]);
  // Derivative of softmax isn't numerically stable.
  const g = grad(f);
  assertAllClose(g([1, 2, 3, 4]), [0, 0, 0, 0]);
});

test(async function api_logSoftmax() {
  const f = (x) => tensor(x).logSoftmax();
  assertAllClose(f([1, 2, 3, 4]),
    [-3.44018984, -2.44018984, -1.44018972, -0.44018975]);
  const g = grad(f);
  assertAllClose(g([1, 2, 3, 4]),
    [0.87176559, 0.65142273, 0.05246873, -1.57565704]);
});

testDevices(async function api_argMaxAndMin(tensor, device) {
  const a = tensor([
    [9, 5, 7],
    [6, 8, 4],
  ]);
  assertAllEqual(a.argmax(1), [0, 1]);
  assertAllEqual(a.argmin(1), [1, 2]);
  assertAllEqual(a.argmax(0), [0, 1, 0]);
  assertAllEqual(a.argmin(0), [1, 0, 1]);
  // Not differentiable.
  const g = grad((x) => tensor(x).argmax(0));
  assertAllEqual(g(a), [
    [0, 0, 0],
    [0, 0, 0],
  ]);
  const h = grad((x) => tensor(x).argmin(0));
  assertAllEqual(h(a), [
    [0, 0, 0],
    [0, 0, 0],
  ]);
});

test(async function api_dot() {
  assertAllEqual(tensor(3).dot(4), 12);
  assertAllEqual(tensor([[3]]).dot([[4]]), [[12]]);
  const r = tensor([9, 5, 7]).dot([6, 8, 4]);
  assertAllEqual(r, 122);
  const m1 = tensor([
    [9, 8, 7],
    [6, 5, 4],
  ]);
  const m2 = tensor([
    [1, 2],
    [4, 5],
    [7, 8],
  ]);
  assertAllEqual(m1.dot(m2), [[90, 114], [54, 69]]);
  assertAllEqual(m1.dot([1, 2, 3]), [46, 28]);
  assertAllEqual(tensor([1, 2, 3]).dot(m2), [30, 36]);
});

test(async function api_zerosOnes() {
  const z1 = zeros([2, 3]);
  assertAllEqual(z1, [[0, 0, 0], [0, 0, 0]]);
  assert(z1.dtype === "float32");

  const z2 = zeros([2, 3], {dtype: "int32"});
  assertAllEqual(z2, [[0, 0, 0], [0, 0, 0]]);
  assert(z2.dtype === "int32");

  const o1 = ones([2, 3]);
  assertAllEqual(o1, [[1, 1, 1], [1, 1, 1]]);
  assert(o1.dtype === "float32");

  const o2 = ones([2, 3], {dtype: "int32"});
  assertAllEqual(o2, [[1, 1, 1], [1, 1, 1]]);
  assert(o2.dtype === "int32");
});

test(async function api_bcastAdd() {
  const a = tensor([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
  ]);
  const b = tensor([42, 43]);
  const f = (x, y) => tensor(x).add(y);
  assertAllEqual(f(a, b), [
    [1 + 42, 2 + 43],
    [3 + 42, 4 + 43],
    [5 + 42, 6 + 43],
    [7 + 42, 8 + 43],
  ]);
  const g = multigrad(f, [0, 1]);
  const gab = g(a, b);
  assert(gab.length === 2);
  assertAllEqual(gab[0], [
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
  ]);
  assertAllEqual(gab[1], [4, 4]);
});

test(async function api_bcastSub() {
  const a = tensor([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
  ]);
  const b = tensor([42, 43]);
  const f = (x, y) => tensor(x).sub(y);
  assertAllEqual(f(a, b), [
    [1 - 42, 2 - 43],
    [3 - 42, 4 - 43],
    [5 - 42, 6 - 43],
    [7 - 42, 8 - 43],
  ]);
  const g = multigrad(f, [0, 1]);
  const gab = g(a, b);
  assert(gab.length === 2);
  assertAllEqual(gab[0], [
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
  ]);
  assertAllEqual(gab[1], [-4, -4]);
});

test(async function api_bcastMul() {
  const a = tensor([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
  ]);
  const b = tensor([42, 43]);
  const f = (x, y) => tensor(x).mul(y);
  assertAllEqual(f(a, b), [
    [1 * 42, 2 * 43],
    [3 * 42, 4 * 43],
    [5 * 42, 6 * 43],
    [7 * 42, 8 * 43],
  ]);
  const g = multigrad(f, [0, 1]);
  const gab = g(a, b);
  assert(gab.length === 2);
  assertAllEqual(gab[0], [
    [42, 43],
    [42, 43],
    [42, 43],
    [42, 43],
  ]);
  assertAllEqual(gab[1], [16, 20]);
});

test(async function api_bcastDiv() {
  const a = tensor([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
  ]);
  const b = tensor([42, 43]);
  const f = (x, y) => tensor(x).div(y);
  assertAllClose(f(a, b), [
    [1 / 42, 2 / 43],
    [3 / 42, 4 / 43],
    [5 / 42, 6 / 43],
    [7 / 42, 8 / 43],
  ]);
  const g = multigrad(f, [0, 1]);
  const gab = g(a, b);
  assert(gab.length === 2);
  assertAllClose(gab[0], [
    [0.02380952, 0.02325581],
    [0.02380952, 0.02325581],
    [0.02380952, 0.02325581],
    [0.02380952, 0.02325581],
  ]);
  assertAllClose(gab[1], [-0.00907029, -0.01081666]);
});

testDevices(async function api_slice(tensor, device) {
  // TODO support uint8 on CUDA.
  const a = tensor([[[1, 1, 1], [2, 2, 2]],
                    [[3, 3, 3], [4, 4, 4]],
                    [[5, 5, 5], [6, 6, 6]]]);
  const s1 = a.slice([1, 0, 0], [1, 1, 3]);
  assertAllEqual(s1, [[[3, 3, 3]]]);
  assertAllEqual(a.slice([1, 0, 0], [1, 2, 3]),
                 [[[3, 3, 3],
                   [4, 4, 4]]]);
  assertAllEqual(a.slice([1, 0, 0], [2, 1, 3]),
                 [[[3, 3, 3]],
                  [[5, 5, 5]]]);
  assertAllEqual(a.slice([1, 0, 0], [1, -1, -1]),
                 [[[3, 3, 3], [4, 4, 4]]]);

  // Convenience APIs.
  const t = tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
  ]);
  assertAllEqual(t.slice(1), [
    [5, 6, 7, 8],
    [9, 10, 11, 12],
  ]);
  assertAllEqual(a.slice(1), [
    [[3, 3, 3], [4, 4, 4]],
    [[5, 5, 5], [6, 6, 6]],
  ]);
  assertAllEqual(a.slice(1, 1), [[[3, 3, 3], [4, 4, 4]]]);

  const s2 = tensor([1, 2, 3], {dtype: "int32"}).slice([1], [1]);
  assert(s2.dtype === "int32");
  assertAllEqual(s2, [2]);
  const f = (x) => tensor(x).slice([1, 0, 0], [2, 1, 3]);
  grad(f);
  // TODO figure out backwards pass.
});

testDevices(async function api_concat(tensor, device) {
  const a = tensor([[[1, 1, 1], [2, 2, 2]],
                    [[3, 3, 3], [4, 4, 4]]]);
  const b = tensor([[[5, 5, 5], [6, 6, 6]],
                    [[7, 7, 7], [8, 8, 8]]]);
  assertAllEqual(a.concat(0, b),
      [[[1, 1, 1], [2, 2, 2]],
       [[3, 3, 3], [4, 4, 4]],
       [[5, 5, 5], [6, 6, 6]],
       [[7, 7, 7], [8, 8, 8]]]);
  assertAllEqual(a.concat(1, b),
      [[[1, 1, 1], [2, 2, 2], [5, 5, 5], [6, 6, 6]],
       [[3, 3, 3], [4, 4, 4], [7, 7, 7], [8, 8, 8]]]);
  assertAllEqual(a.concat(2, b),
      [[[1, 1, 1, 5, 5, 5],
        [2, 2, 2, 6, 6, 6]],
       [[3, 3, 3, 7, 7, 7],
        [4, 4, 4, 8, 8, 8]]]);
  assertAllEqual(a.concat(2, b, b),
      [[[1, 1, 1, 5, 5, 5, 5, 5, 5],
        [2, 2, 2, 6, 6, 6, 6, 6, 6]],
       [[3, 3, 3, 7, 7, 7, 7, 7, 7],
        [4, 4, 4, 8, 8, 8, 8, 8, 8]]]);
  const t = tensor([[1, 2], [3, 4]]);
  const s = t.concat(0, [[5, 6]]);
  assertAllEqual(s, [[1, 2], [3, 4], [5, 6]]);
  // Backwards pass.
  const f = (x, y) => x.concat(0, y.mul(2));
  const g = multigrad(f, [0, 1]);
  const gab = g(a, b);
  assertAllEqual(gab[0], [[[1, 1, 1], [1, 1, 1]],
                          [[1, 1, 1], [1, 1, 1]]]);
  assertAllEqual(gab[1], [[[2, 2, 2], [2, 2, 2]],
                          [[2, 2, 2], [2, 2, 2]]]);
});

test(async function api_cast() {
  const a = tensor([255, 127, 0], {dtype: "uint8"});
  assert(a.dtype === "uint8");
  const r = a.cast("float32").div(255);
  assertAllClose(r, [1.0, 127 / 255, 0]);
});

testDevices(async function api_oneHot(tensor, device) {
  const a = tensor([0, 1, 3, 4], {dtype: "int32"});
  assertAllEqual(a.oneHot(6), [
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
  ]);

  const b = tensor([0, 1, 3, 4], {dtype: "int32"});
  assertAllEqual(b.oneHot(5, 0.5, -0.5), [
    [ 0.5, -0.5, -0.5, -0.5, -0.5],
    [-0.5,  0.5, -0.5, -0.5, -0.5],
    [-0.5, -0.5, -0.5,  0.5, -0.5],
    [-0.5, -0.5, -0.5, -0.5,  0.5],
  ]);
  // Make sure it works with uint8.
  const c = tensor([0, 1], {dtype: "uint8"});
  assertAllEqual(c.oneHot(3), [[1, 0, 0], [0, 1, 0]]);
});

test(async function api_softmaxCE() {
  const labels = tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.3, 0.0, 0.7],
  ], {dtype: "float32"});
  const f = (x) => tensor(x).softmaxCE(labels);
  const logits = tensor([
    [-2, 2, 10],
    [-2, 2, 10],
    [-2, 2, 10],
  ], {dtype: "float32"});
  const ce = f(logits);
  assertAllClose(ce, [12.00034142, 8.00034142, 3.6003418]);
  const g = grad(f);
  assertAllClose(g(logits), [
    [ -9.99993861e-01,   3.35348042e-04,   9.99658465e-01],
    [  6.14211376e-06,  -9.99664664e-01,   9.99658465e-01],
    [ -2.99993873e-01,   3.35348042e-04,   2.99658477e-01]
  ]);
});

test(async function api_softmaxLoss() {
  const logits = tensor([
    [-2, 2, 10],
    [-3, 2, 10],
  ], {dtype: "float32"});
  const loss = logits.softmaxLoss([0, 2]);
  assertAllClose(loss, 6.0);
});

test(async function api_devicePlacement() {
  if (!gpuAvail()) {
    console.log("GPU not available. Skipping testDevicePlacement.");
    return;
  }
  const t = tensor([1, 2]).gpu();
  const s = tensor([3, 4]).gpu();
  const r = t.mul(s);
  const rCpu = r.cpu();
  assert(t.device === "GPU:0");
  assert(s.device === "GPU:0");
  assert(r.device === "GPU:0");
  assert(rCpu.device === "CPU:0");
  assertAllEqual(rCpu, [3, 8]);
});

testDevices(async function api_neuralNet(tensor, device) {
  const inference = (params: Params, images: Tensor) => {
    let inputs = images.cast("float32").div(255).reshape([-1, 28 * 28]);
    let outputs;
    const layerSizes = [ 28 * 28, 64, 10 ];
    for (let i = 0; i < layerSizes.length - 1; ++i) {
      const m = layerSizes[i];
      const n = layerSizes[i + 1];
      // Initialize or get weights and biases.
      const w = params.define(`w${i}`, () =>
        randn([m, n], { dtype: "float32", device }));
      const b = params.define(`b${i}`, () =>
        zeros([n], { dtype: "float32", device }));
      outputs = inputs.matmul(w).add(b);
      inputs = outputs.relu();
    }
    return outputs;
  };

  // Define the training objective using softmax cross entropy loss.
  const loss = (images, labels, params: Params): Tensor => {
    const labels1H = labels.oneHot(10);
    const logits = inference(params, images);
    const softmaxLoss = logits.softmaxCE(labels1H).reduceMean();
    return softmaxLoss;
  };

  // Just zero data.
  const images = tensor(zeros([16, 28, 28], {dtype: "int32"}));
  const labels = tensor(zeros([16], {dtype: "int32"}));
  let params = api.params();
  const gradFn = api.gradParams((params: Params): Tensor => {
    return loss(images, labels, params);
  });
  const steps = 3;
  const learningRate = 0.001;
  for (let i = 0; i < steps; i++) {
    const [grads] = gradFn(params);
    const updated = api.params();
    for (const name of Object.keys(grads)) {
      const g = grads[name];
      const p = params.get(name);
      if (i > 0) {
        assertShapesEqual(p.shape, g.shape);
      }
      updated.set(name, p.sub(g.mul(learningRate)));
    }
    params = updated;
  }
});

testDevices(async function api_setDiag(tensor, device) {
  const matrix = tensor(zeros([4, 3]));
  const m = matrix.setDiag([1, 2, 3]);
  assertAllEqual(m, [
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 3],
    [0, 0, 0]
  ]);
});

testDevices(async function api_eye(tensor, device) {
  const eye4 = tensor(api.eye(4));
  assertAllEqual(eye4, [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]);
});

if (IS_NODE) {
  test(async function api_inspect() {
    const t = tensor([1, 2, 3]);
    const actual = require("util").inspect(t);
    assert("[ 1.,  2.,  3.]" === actual);
  });
}

test(async function api_plotSmoke() {
  // Just make sure we can import and call some matplotlib function without
  // crashing.
  const x = tensor([1, 2, 3]);
  const y = tensor([4, 5, 6]);
  api.plot(x, y);
  const im = zeros([9, 9]);
  api.imshow(im);
});

test(async function api_rangeIterable() {
  const items = [];
  for (const i of range(3)) {
    items.push(i);
  }
  assertAllEqual(items, [0, 1, 2]);
});

test(async function api_rescale() {
  const t = tensor([0, 50, 255]);
  const r = t.rescale([0, 255], [-1, 1]);
  assertAllClose(r, [-1, -0.607843137254902, 1]);
});

test(async function api_data() {
  const t = api.range(3);
  assertAllEqual(await t.data(), [0, 1, 2]);
  assertAllEqual(t.dataSync(), [0, 1, 2]);
});

test(async function api_dtypeConstructors() {
  const ui8 = api.uint8([254, 255, 256, 257]);
  assert(ui8.dtype === "uint8");
  assertAllEqual(ui8, [254, 255, 0, 1]);
  const i32 = api.int32([1, 2, 3]);
  assert(i32.dtype === "int32");
  assertAllEqual(i32, [1, 2, 3]);
  const f32 = api.float32([1.5, 3.5]);
  assert(f32.dtype === "float32");
  assertAllEqual(f32, [1.5, 3.5]);
});

test(async function api_conv2d() {
  // Just check that conv2d works with grad, more detailed
  // tests are done in src/conv_test.ts
  const f = (a, b) => api.conv2d(a, b);
  const g = multigrad(f, [0, 1]);
  const img = api.range(4 * 4).reshape([1, 4, 4, 1]).cast("float32");
  const filter = api.range(2 * 2).reshape([2, 2, 1, 1]).cast("float32");
  const g_ = g(img, filter);
  assertShapesEqual(g_[0].shape, img.shape);
  assertShapesEqual(g_[1].shape, filter.shape);
  assertAllEqual(g_[0].squeeze(), [
    [ 0, 1, 1, 1 ],
    [ 2, 6, 6, 4 ],
    [ 2, 6, 6, 4 ],
    [ 2, 5, 5, 3 ],
  ]);
  assertAllEqual(g_[1].squeeze(), [[45, 54], [81, 90]]);
});

test(async function api_maxPool() {
  // TODO Port over TF test cases like was done in src/conv_testcases.ts
  // tslint:disable-next-line:max-line-length
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/kernel_tests/pooling_ops_test.py#L227-L710
  const x = api.range(4 * 4).reshape([1, 4, 4, 1]);
  const r = x.maxPool({ size: 2, stride: 2 });
  assertShapesEqual(r.shape, [1, 2, 2, 1]);
  assertAllEqual(r.squeeze(), [[5, 7], [13, 15]]);
  const f = img => img.maxPool();
  const g = grad(f);
  const gx = g(x);
  assertShapesEqual(gx.shape, x.shape);
  assertAllEqual(gx.squeeze(), [
    [ 0, 0, 0, 0 ],
    [ 0, 1, 0, 1 ],
    [ 0, 0, 0, 0 ],
    [ 0, 1, 0, 1 ],
  ]);
});

test(async function api_moments() {
  const x = api.range(4).reshape([2, 2]);
  const { mean, variance } = x.moments([1]);
  assertAllEqual(mean, [0.5, 2.5]);
  assertAllEqual(variance, [0.25, 0.25]);
});

test(async function api_sin_cos_tan() {
  const sin = x => tensor(x).sin();
  const cos = x => tensor(x).cos();
  const tan = x => tensor(x).tan();
  const v = tensor([0, Math.PI / 4, Math.PI]);
  const a = 1 / Math.sqrt(2);
  assertAllClose(sin(v), [0, a, 0]);
  assertAllClose(cos(v), [1, a, -1]);
  assertAllClose(tan(v), [0, 1, 0]);
  assertAllClose(grad(sin)(v), [1, a, -1]);
  assertAllClose(grad(cos)(v), [0, a, 0]);
  assertAllClose(grad(tan)(v), [1, 2, 1]);
});
