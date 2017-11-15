import $ from './propel';
import { assertEqual, assertAllEqual } from './util';

function testShapes() {
  let a = $([[1, 2], [3, 4]]);
  assertAllEqual(a.shape, [2, 2]);

  let b = $([[[1, 2]]]);
  assertAllEqual(b.shape, [1, 1, 2]);
}

function testMul() {
  let a = $([[1, 2], [3, 4]]);
  let expected = $([[1, 4], [9, 16]]);
  let actual = a.mul(a);
  assertAllEqual(actual.shape, [2, 2]);
  assertAllEqual(actual, expected);
}

testShapes();
testMul();
