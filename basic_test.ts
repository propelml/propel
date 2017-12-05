import { convertBasic, basicOps } from "./basic";
import { assertShapesEqual, assertAllEqual } from "./util";

const $ = convertBasic;

function testShapes() {
  const a = $([[1, 2], [3, 4]]);
  assertShapesEqual(a.shape, [2, 2]);

  const b = $([[[1, 2]]]);
  assertShapesEqual(b.shape, [1, 1, 2]);

  assertShapesEqual($(42).shape, []);
  assertShapesEqual($([42]).shape, [1]);
}

function testMul() {
  const a = $([[1, 2], [3, 4]]);
  const expected = $([[1, 4], [9, 16]]);
  const actual = basicOps.mul(a, a);
  assertAllEqual(actual.shape, [2, 2]);
  assertAllEqual(actual, expected);
}

testShapes();
testMul();
