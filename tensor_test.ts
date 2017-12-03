import { $, allEqual, linspace, arange, concat, stack } from "./propel";
import { assert, assertFalse, assertShapesEqual, assertEqual, assertAllEqual,
  assertAllClose } from "./util";

function testShapes() {
  const a = $([[1, 2], [3, 4]]);
  assertShapesEqual(a.shape, [2, 2]);

  const b = $([[[1, 2]]]);
  assertShapesEqual(b.shape, [1, 1, 2]);

  assertShapesEqual($(42).shape, []);
  assertShapesEqual($([42]).shape, [1]);
  assertShapesEqual($([]).shape, []);
}

function testAllEqual() {
  assert(allEqual([1, 2, 3], [1, 2, 3]));
  assert(allEqual([[1], [2]], [[1], [2]]));
  const t = $([[1], [3]]);
  const s = $([[1], [2]]);
  assertFalse(t.equals(s));
  assertFalse(allEqual(s, t));
  assert(allEqual([], []));
  assertFalse($(0).equals([]));
}

function testLinspace() {
  const x = linspace(-4, 4, 6);
  assertAllClose(x, [-4., -2.4, -0.8,  0.8,  2.4, 4.]);
}

function testMul() {
  const a = $([[1, 2], [3, 4]]);
  const expected = $([[1, 4], [9, 16]]);
  const actual = a.mul(a);
  assertAllEqual(actual.shape, [2, 2]);
  assertAllEqual(actual, expected);
}

function testReshape() {
  const x = arange(0, 6).reshape([2, 3]);
  assertAllEqual(x.shape, [2, 3]);
  assertEqual(x.get(0, 0), 0);
  assertEqual(x.get(0, 1), 1);
  assertEqual(x.get(0, 2), 2);
  assertEqual(x.get(1, 0), 3);
  assertEqual(x.get(1, 1), 4);
  assertEqual(x.get(1, 2), 5);
}

function testExpandDims() {
  const x = arange(0, 6).reshape([2, 3]);
  const y = x.expandDims(1);
  const z = y.expandDims(0);
  assertAllEqual(x.shape, [2, 3]);
  assertAllEqual(y.shape, [2, 1, 3]);
  assertAllEqual(z.shape, [1, 2, 1, 3]);
}

function testConcat() {
  const x = arange(0, 6).reshape([2, 3]);
  const y = arange(6, 12).reshape([2, 3]);
  const r = concat([x, y], 1);
  assertShapesEqual(r.shape, [2, 6]);
  assertEqual(r.get(0, 0), 0);
  assertEqual(r.get(0, 3), 6);
  assertEqual(r.get(0, 4), 7);
  assertEqual(r.get(1, 4), 10);
}

function testStack() {
  const x = arange(0, 6).reshape([2, 3]);
  const y = arange(6, 12).reshape([2, 3]);
  const r = stack([x, y], 0);
  assertShapesEqual(r.shape, [2, 2, 3]);
  assertEqual(r.get(0, 0, 0), 0);
  assertEqual(r.get(1, 0, 0), 6);
}

testShapes();
testAllEqual();
testMul();
testReshape();
testExpandDims();
testLinspace();
testConcat();
testStack();
