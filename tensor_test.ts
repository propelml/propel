import * as sp from './sigprop';
import {assertEqual, assertAllEqual} from './util';

function testShapes() {
  let a = new sp.Tensor([[1, 2], [3, 4]]);
  assertAllEqual(a.shape, [2, 2]);

  let b = new sp.Tensor([[[1, 2]]]);
  assertAllEqual(b.shape, [1, 1, 2]);
}

function testMul() {
  let a = new sp.Tensor([[1, 2], [3, 4]]);
  let expected = new sp.Tensor([[1, 4], [9, 16]]);
  let actual = sp.mul(a, a);
  assertAllEqual(actual.shape, [2, 2]);
  assertAllEqual(actual, expected);
}

testShapes();
testMul();
