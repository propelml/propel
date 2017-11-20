import $ from './propel';
import { assert, assertFalse, assertShapesEqual, assertEqual, assertAllEqual } from './util';

function testShapes() {
  let a = $([[1, 2], [3, 4]]);
  assertShapesEqual(a.shape, [2, 2]);
  let b = $([[[1, 2]]]);
  assertShapesEqual(b.shape, [1, 1, 2]);
  assertShapesEqual($(42).shape, []);
  assertShapesEqual($([42]).shape, [1]);
  assertShapesEqual($([]).shape, []);
}

function testAllEqual() {
  assert($.allEqual([1, 2, 3], [1, 2, 3]));
  assert($.allEqual([[1], [2]], [[1], [2]]));
  let t = $([[1], [3]]);
  let s = $([[1], [2]]);
  assertFalse(t.equals(s));
  assertFalse($.allEqual(s, t));
  assert($.allEqual([], []));
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

function testConcat() {
  let x = $.arange(0, 6).reshape([2, 3]);
  let y = $.arange(6, 12).reshape([2, 3]);
  let r = $.concat([x, y], 1);
  assertShapesEqual(r.shape, [2, 6]);
  assertEqual(r.get(0, 0), 0);
  assertEqual(r.get(0, 3), 6);
  assertEqual(r.get(0, 4), 7);
  assertEqual(r.get(1, 4), 10);
}

function testStack() {
  let x = $.arange(0, 6).reshape([2, 3]);
  let y = $.arange(6, 12).reshape([2, 3]);
  let r = $.stack([x, y], 0);
  assertShapesEqual(r.shape, [2, 2, 3]);
  assertEqual(r.get(0, 0, 0), 0);
  assertEqual(r.get(1, 0, 0), 6);
}

testShapes();
testAllEqual();
testMul();
testReshape();
testExpandDims();
testConcat();
testStack();
