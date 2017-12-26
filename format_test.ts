import * as format from "./format";
import { assert } from "./util";

function test1DInt32() {
  const actual = format.toString([4], new Int32Array([1, 2, 3, 4]));
  const expected = "[ 1, 2, 3, 4 ]";
  assert(actual === expected);
  console.log(actual);
}

function test2DInt32() {
  const actual = format.toString([2, 2], new Int32Array([1, 2, 3, 4]));
  const expected = "[ [ 1, 2 ],\n  [ 3, 4 ] ]";
  assert(actual === expected);
  console.log(actual);
}

test1DInt32();
test2DInt32();
