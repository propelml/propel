import sp from './sigprop';
import {assertClose, log} from './util';

function checkGrad(val, f, fGrad) {
  let epsilon = 0.0001;
  let expected = f(val + epsilon).sub(f(val - epsilon)).div(2 * epsilon);
  let actual = fGrad(val)[0];
  assertClose(actual, expected);
}

function testInc() {
  function inc(x) {
    return sp(x).add(1);
  }

  assertClose(inc(1), 2);
  assertClose(inc(-1), 0);

  let gradInc = sp.grad(inc);
  let r = gradInc(1.0)[0];
  assertClose(r, 1.);
}

function testMul() {
  function f(x) {
    return sp(42).mul(x);
  }

  assertClose(f(1), 42);
  assertClose(f(2), 84);

  let g = sp.grad(f);
  let r = g(1.0)[0];
  assertClose(r, 42.);
}

function testSquared() {
  // f(x) = x^2
  function f(x) {
    return sp(x).mul(x);
  }
  assertClose(f(1), 1);
  assertClose(f(16), 256);
  let g = sp.grad(f); // g(x) = f'(x) = 2x

  assertClose(g(1)[0], 2);
  assertClose(g(10)[0], 20);
}

function testDiv() {
  // f(x) = (1 + x) / x
  function f(x) {
    x = sp(x);
    return x.add(1).div(x);
  }
  assertClose(f(1), 2);
  assertClose(f(16), (1 + 16) / 16);
  let g = sp.grad(f); // g(x) = -1 / x^2

  assertClose(g(1)[0], -1);
  assertClose(g(10)[0], -1 / 100);
}

function testConstant() {
  let c = (_) => 42;
  assertClose(c(1), 42);
  assertClose(c(-1), 42);
  let g = sp.grad(c);
  assertClose(g(1.0)[0], 0.);
}

function testExp() {
  // f(x) = exp(1+x)
  function f(x) {
    return sp(x).add(1).exp();
  }
  assertClose(f(1), 7.3890);
  assertClose(f(2), 20.0855);
  let g = sp.grad(f); // g == f
  assertClose(g(1)[0], 7.3890);
  assertClose(g(2)[0], 20.0855);
}

function testSub() {
  function f(x) {
    return sp(1).sub(x);
  }
  assertClose(f(1), 0);
  assertClose(f(2), -1);
  let g = sp.grad(f);
  assertClose(g(1)[0], -1);
  assertClose(g(2)[0], -1);
}

function testDiv2() {
  function f(x) {
    x = sp(x);
    return sp(1).sub(x).div(x.add(1)); 
  }
  assertClose(f(1), 0);
  assertClose(f(2), -1/3);
  let g = sp.grad(f); // g(x) = -2 / (x + 1)^2
  assertClose(g(1)[0], -2/4);
  assertClose(g(2)[0], -2/9);
}

function testDiv3() {
  function f(x) {
    let y = sp(x).exp();
    return y.div(y);
  }
  assertClose(f(1), 1.);
  assertClose(f(2), 1.);
  let g = sp.grad(f);

  assertClose(g(1)[0], 0.);
  assertClose(g(2)[0], 0.);
}

function testTanh() {
  function tanh(x) {
    let y = sp(x).mul(-2).exp();
    return sp(1).sub(y).div(y.add(1));
  }

  assertClose(tanh(1), 0.7615);
  assertClose(tanh(16), 0.9999);

  let gradTanh = sp.grad(tanh);
  assertClose(gradTanh(1)[0], 0.4199);
}

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
