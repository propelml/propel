import { $, grad, tanh, multigrad, linspace, arange } from "./api";
import { assertClose, assertAllEqual, assertAllClose} from "./util";

function checkGrad(f, g, val = 1.0) {
  const epsilon = 0.01;
  const a = $(f(val + epsilon));
  const b = $(f(val - epsilon));
  const expected = a.sub(b).div(2 * epsilon);
  const actual = g(val);
  assertClose(actual, expected);
}

// Basic Tests

function testLinspace() {
  const x = linspace(-4, 4, 6);
  assertAllClose(x, [-4., -2.4, -0.8,  0.8,  2.4, 4.]);
}

function testArange() {
  const x = arange(-2, 2, 1);
  assertAllEqual(x, [-2, -1, 0, 1]);
}


// Backprop Tests

function testInc() {
  function f(x) {
    return $(x).add(1);
  }
  assertClose(f(1), 2);
  assertClose(f(-1), 0);
  const g = grad(f);
  assertClose(g(1.0), 1.);
  checkGrad(f, g, 1.0);
}

function testMul() {
  const f = (x) => $(42).mul(x);
  assertClose(f(1), 42);
  assertClose(f(2), 84);
  const g = grad(f);
  assertClose(g(1.), 42.);
  checkGrad(f, g, 1.0);
}

function testSquared() {
  // f(x) = x^2
  function f(x) {
    return $(x).mul(x);
  }
  assertClose(f(1), 1);
  assertClose(f(16), 256);
  const g = grad(f); // g(x) = f'(x) = 2x
  assertClose(g(1), 2);
  assertClose(g(10), 20);
  checkGrad(f, g, 1.0);
}

function testSquaredMatrix() {
  // f(x) = x^2
  function f(x) {
    return $(x).mul(x);
  }
  assertAllEqual(f([[1, 2], [3, 4]]), [[1, 4], [9, 16]]);
  const g = grad(f); // g(x) = f'(x) = 2x
  const v = g([[1, 2], [3, 4]]);
  assertAllEqual(v.shape, [2, 2]);
  assertAllEqual(v, [[2, 4], [6, 8]]);
}

function testDiv() {
  // f(x) = (1 + x) / x
  function f(x) {
    x = $(x);
    return x.add(1).div(x);
  }
  assertClose(f(1), 2);
  assertClose(f(16), (1 + 16) / 16);
  const g = grad(f); // g(x) = -1 / x^2
  assertClose(g(1), -1);
  assertClose(g(10), -1 / 100);
  checkGrad(f, g, 1.0);
}

function testConstant() {
  const f = (_) => 42;
  assertClose(f(1), 42);
  assertClose(f(-1), 42);
  const g = grad(f);
  assertClose(g(1.0), 0.);
  checkGrad(f, g, 1.0);
}

function testExp() {
  // f(x) = exp(1+x)
  function f(x) {
    return $(x).add(1).exp();
  }
  assertClose(f(1), 7.3890);
  assertClose(f(2), 20.0855);
  const g = grad(f); // g == f
  assertClose(g(1), 7.3890);
  assertClose(g(2), 20.0855);
  checkGrad(f, g, 1.0);
}

function testSub() {
  function f(x) {
    return $(1).sub(x);
  }
  assertClose(f(1), 0);
  assertClose(f(2), -1);
  const g = grad(f);
  assertClose(g(1), -1);
  assertClose(g(2), -1);
  checkGrad(f, g, 1.0);
}

function testDiv2() {
  function f(x) {
    x = $(x);
    return $(1).sub(x).div(x.add(1));
  }
  assertClose(f(1), 0);
  assertClose(f(2), -1 / 3);
  const g = grad(f); // g(x) = -2 / (x + 1)^2
  assertClose(g(1), -2 / 4);
  assertClose(g(2), -2 / 9);
  checkGrad(f, g, 1.0);
}

function testDiv3() {
  function f(x) {
    const y = $(x).exp();
    return y.div(y);
  }
  assertClose(f(1), 1.);
  assertClose(f(2), 1.);
  const g = grad(f);
  assertClose(g(1), 0.);
  assertClose(g(2), 0.);
  checkGrad(f, g, 1.0);
}

function testTanh() {
  const f = tanh;
  assertClose(f(1), 0.7615);
  assertClose(f(16), 0.9999);
  const g = grad(f);
  assertClose(g(1), 0.4199);
  checkGrad(f, g, 1.0);
}

function testMultigrad() {
  function f(a, b) {
    return $(a).mul(2).add($(b).mul(3));
  }
  assertClose(f(1, 1), 5);
  assertClose(f(1, 2), 8);
  const g = multigrad(f, [0, 1]);
  assertClose(g(1, 1)[0], 2);
  assertClose(g(1, 1)[1], 3);
  assertClose(g(4, 2)[0], 2);
  assertClose(g(4, 2)[1], 3);
}

function testGradGradTanh() {
  const f = tanh;
  assertAllClose(f([1, 16]), [0.7615, 0.9999]);
  const g = grad(grad(f));
  // def g(x): return -2 * np.tanh(x) / np.square(np.cosh(x))
  assertAllClose(g([1, 2]), [-0.6397, -0.13621]);
}


testLinspace();
testArange();

testInc();
testMul();
testSquared();
testDiv();
testConstant();
testExp();
testSub();
testDiv2();
testDiv3();
testTanh();
testMultigrad();
testSquaredMatrix();
testGradGradTanh();

console.log("PASS");
