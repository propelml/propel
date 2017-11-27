import $ from "./propel";
import { assertAllEqual, assertEqual, assertShapesEqual } from "./util";

function testShapes() {
  const a = $([[1, 2], [3, 4]]);
  assertAllEqual(a.shape, [2, 2]);

  const b = $([[[1, 2]]]);
  assertAllEqual(b.shape, [1, 1, 2]);

  assertShapesEqual($(42).shape, []);
  assertShapesEqual($([42]).shape, [1]);
}

function testLinspace() {
  const x = $.linspace(-4, 4, 6);
  assertAllEqual(x, [-4., -2.4, -0.8,  0.8,  2.4, 4.]);
}

function testMul() {
  const a = $([[1, 2], [3, 4]]);
  const expected = $([[1, 4], [9, 16]]);
  const actual = a.mul(a);
  assertAllEqual(actual.shape, [2, 2]);
  assertAllEqual(actual, expected);
}

function testReshape() {
  const x = $.arange(0, 6).reshape([2, 3]);
  assertAllEqual(x.shape, [2, 3]);
  assertEqual(x.get(0, 0), 0);
  assertEqual(x.get(0, 1), 1);
  assertEqual(x.get(0, 2), 2);
  assertEqual(x.get(1, 0), 3);
  assertEqual(x.get(1, 1), 4);
  assertEqual(x.get(1, 2), 5);
}

function testExpandDims() {
  const x = $.arange(0, 6).reshape([2, 3]);
  const y = x.expandDims(1);
  const z = y.expandDims(0);
  assertAllEqual(x.shape, [2, 3]);
  assertAllEqual(y.shape, [2, 1, 3]);
  assertAllEqual(z.shape, [1, 2, 1, 3]);
}

testShapes();
testMul();
testReshape();
testExpandDims();
testLinspace();
