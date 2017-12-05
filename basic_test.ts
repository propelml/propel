import { convertBasic, basicOps } from "./basic";
import { assertShapesEqual, assertAllEqual, assertAllClose } from "./util";

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

function testSquare() {
  const a = $([[1, 2], [3, 4]]);
  const expected = $([[1, 4], [9, 16]]);
  const actual = basicOps.square(a);
  assertAllEqual(actual.shape, [2, 2]);
  assertAllEqual(actual, expected);
}

function testCosh() {
  const a = $([[1, 2], [3, 4]]);
  const actual = basicOps.cosh(a);
  assertAllEqual(actual.shape, [2, 2]);
  const expected = [
    [  1.54308063,   3.76219569],
    [ 10.067662  ,  27.30823284],
  ];
  assertAllClose(actual, expected);
}

testShapes();
testMul();
testSquare();
testCosh();
