import * as util from "./util";
import { assert, assertEqual, assertShapesEqual } from "./util";

function testCounterMap() {
  const m = new util.CounterMap();
  assertEqual(m.get(0), 0);
  m.inc(0);
  m.inc(0);
  assertEqual(m.get(0), 2);
  assertEqual(m.get(1), 0);
  m.inc(1);
  assertEqual(m.get(1), 1);
  m.dec(0);
  m.inc(1);
  assertEqual(m.get(0), 1);
  assertEqual(m.get(1), 2);
  const k = m.keys();
  console.log(k);
}

function testDeepCloneArray() {
  const arr1 = [[1, 2], [3, 4]];
  const arr2 = util.deepCloneArray(arr1);
  // Verify that arrays have different identity.
  assert(arr1 !== arr2);
  assert(arr1[0] !== arr2[0]);
  assert(arr1[1] !== arr2[1]);
  // Verify that primitives inside the array have the same value.
  assert(arr1[0][0] === arr2[0][0]);
  assert(arr1[1][0] === arr2[1][0]);
  assert(arr1[0][1] === arr2[0][1]);
  assert(arr1[1][1] === arr2[1][1]);
}

function testBcastGradientArgs() {
  let [r0, r1] = util.bcastGradientArgs([2, 3, 5], [1]);
  assertShapesEqual(r0, []);
  assertShapesEqual(r1, [0, 1, 2]);

  [r0, r1] = util.bcastGradientArgs([1], [2, 3, 5]);
  assertShapesEqual(r0, [0, 1, 2]);
  assertShapesEqual(r1, []);

  [r0, r1] = util.bcastGradientArgs([2, 3, 5], [5]);
  assertShapesEqual(r0, []);
  assertShapesEqual(r1, [0, 1]);

  [r0, r1] = util.bcastGradientArgs([5], [2, 3, 5]);
  assertShapesEqual(r0, [0, 1]);
  assertShapesEqual(r1, []);

  [r0, r1] = util.bcastGradientArgs([2, 3, 5], [3, 5]);
  assertShapesEqual(r0, []);
  assertShapesEqual(r1, [0]);

  [r0, r1] = util.bcastGradientArgs([3, 5], [2, 3, 5]);
  assertShapesEqual(r0, [0]);
  assertShapesEqual(r1, []);

  [r0, r1] = util.bcastGradientArgs([2, 3, 5], [3, 1]);
  assertShapesEqual(r0, []);
  assertShapesEqual(r1, [0, 2]);

  [r0, r1] = util.bcastGradientArgs([3, 1], [2, 3, 5]);
  assertShapesEqual(r0, [0, 2]);
  assertShapesEqual(r1, []);

  [r0, r1] = util.bcastGradientArgs([2, 1, 5], [3, 1]);
  assertShapesEqual(r0, [1]);
  assertShapesEqual(r1, [0, 2]);

  [r0, r1] = util.bcastGradientArgs([3, 1], [2, 1, 5]);
  assertShapesEqual(r0, [0, 2]);
  assertShapesEqual(r1, [1]);
}

testCounterMap();
testDeepCloneArray();
testBcastGradientArgs();
