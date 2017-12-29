import { toString } from "./format";
import { test } from "./test";
import { assert } from "./util";

test(function test1DInt32() {
  const actual = toString([4], new Int32Array([1, 2, 3, 4]));
  const expected = "[1, 2, 3, 4]";
  console.log(actual);
  assert(actual === expected);
});

test(function test2DInt32() {
  const actual = toString([2, 2], new Int32Array([1, 2, 3, 4]));
  const expected = "[[1, 2],\n [3, 4]]";
  console.log(actual);
  assert(actual === expected);
});

test(function test1DFloat32() {
  const d = [ 1.2,  0. ,  0. ,  0.123];
  const actual = toString([4], new Float32Array(d));
  const expected = "[ 1.2  ,  0.   ,  0.   ,  0.123]";
  console.log(actual);
  assert(actual === expected);
});
test(function test2DFloat32() {
  let actual = toString([2, 2], new Float32Array([1, 2, 3, 4]));
  let expected = "[[ 1.,  2.],\n [ 3.,  4.]]";
  console.log(actual);
  assert(actual === expected);

  const ta = new Float32Array([1.2, 0, 0, 0.1, 1.2, 0, 0.5, 0]);
  actual = toString([2, 4], ta);
  expected = "[[ 1.2,  0. ,  0. ,  0.1],\n [ 1.2,  0. ,  0.5,  0. ]]";
  console.log(actual);
  assert(actual === expected);
});
