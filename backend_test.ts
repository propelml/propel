import { bo, convertBasic } from "./backend";
import { test } from "./test";
import { assertAllClose, assertAllEqual, assertShapesEqual } from "./util";

const $ = convertBasic;

test(function testShapes() {
  const a = $([[1, 2], [3, 4]]);
  assertShapesEqual(a.shape, [2, 2]);

  const b = $([[[1, 2]]]);
  assertShapesEqual(b.shape, [1, 1, 2]);

  assertShapesEqual($(42).shape, []);
  assertShapesEqual($([42]).shape, [1]);
});

test(function testMul() {
  const a = $([[1, 2], [3, 4]]);
  const expected = $([[1, 4], [9, 16]]);
  const actual = bo.mul(a, a);
  assertAllEqual(actual.shape, [2, 2]);
  assertAllEqual(actual, expected);
});

test(function testSquare() {
  const a = $([[1, 2], [3, 4]]);
  const expected = $([[1, 4], [9, 16]]);
  const actual = bo.square(a);
  assertAllEqual(actual.shape, [2, 2]);
  assertAllEqual(actual, expected);
});

test(function testCosh() {
  const a = $([[1, 2], [3, 4]]);
  const actual = bo.cosh(a);
  assertAllEqual(actual.shape, [2, 2]);
  const expected = [
    [  1.54308063,   3.76219569],
    [ 10.067662  ,  27.30823284],
  ];
  assertAllClose(actual, expected);
});

test(function testReluGrad() {
  const grad = $([[-1, 42], [-3, 4]]);
  const ans = $([[-7, 4], [ 0.1, -9 ]]);
  const actual = bo.reluGrad(grad, ans);
  assertAllEqual(actual.shape, [2, 2]);
  assertAllClose(actual, [[0, 42], [-3,  0]]);
});
