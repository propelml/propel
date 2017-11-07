import sp from './sigprop';
import {assertEqual, assertAllEqual} from './util';

function testShapes() {
  let a = sp([[1, 2], [3, 4]]);
  assertAllEqual(a.shape, [2, 2]);

  let b = sp([[[1, 2]]]);
  assertAllEqual(b.shape, [1, 1, 2]);
}

function testMul() {
  let a = sp([[1, 2], [3, 4]]);
  let expected = sp([[1, 4], [9, 16]]);
  let actual = a.mul(a);
  assertAllEqual(actual.shape, [2, 2]);
  assertAllEqual(actual, expected);
}

testShapes();
testMul();
