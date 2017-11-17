import $ from './propel';
import { assertShapesEqual, assertEqual, assertAllEqual } from './util';

function testShapes() {
  let a = $([[1, 2], [3, 4]]);
  assertAllEqual(a.shape, [2, 2]);

  let b = $([[[1, 2]]]);
  assertAllEqual(b.shape, [1, 1, 2]);

  assertShapesEqual($(42).shape, []);
  assertShapesEqual($([42]).shape, [1]);
}

function testMul() {
  let a = $([[1, 2], [3, 4]]);
  let expected = $([[1, 4], [9, 16]]);
  let actual = a.mul(a);
  assertAllEqual(actual.shape, [2, 2]);
  assertAllEqual(actual, expected);
}

function testReshape() {
  let x = $.arange(0, 6).reshape([2, 3]);
  assertAllEqual(x.shape, [2, 3]);
  assertEqual(x.get(0, 0), 0);
  assertEqual(x.get(0, 1), 1);
  assertEqual(x.get(0, 2), 2);
  assertEqual(x.get(1, 0), 3);
  assertEqual(x.get(1, 1), 4);
  assertEqual(x.get(1, 2), 5);
}

function testExpandDims() {
  let x = $.arange(0, 6).reshape([2, 3]);
  let y = x.expandDims(1);
  let z = y.expandDims(0);
  assertAllEqual(x.shape, [2, 3]);
  assertAllEqual(y.shape, [2, 1, 3]);
  assertAllEqual(z.shape, [1, 2, 1, 3]);
}

testShapes();
testMul();
testReshape();
testExpandDims();
