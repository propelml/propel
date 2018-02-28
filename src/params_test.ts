import { test } from "../tools/tester";
import { randn, zeros } from "./api";
import { params } from "./params";
import { assert, assertAllEqual, assertShapesEqual } from "./tensor_util";

test(async function params_smoke() {
  const p = params();
  // randn
  const a = p.define("a", () => randn([2, 3]));
  assert(a.dataSync()[0] !== 0);
  assertShapesEqual(a.shape, [2, 3]);
  const aa = p.define("a", () => randn([2, 3]));
  assertAllEqual(a, aa);
  // zeros
  const b = p.define("b", () => zeros([2, 3]));
  assertAllEqual(b, [[0, 0, 0], [0, 0, 0]]);
  const bb = p.define("b", () => zeros([2, 3]));
  assertAllEqual(b, bb);
});

test(async function params_scope() {
  const p = params();
  const paramsL1 = p.scope("L1");
  const w = paramsL1.define("weights", () => randn([2, 3]));
  const b = paramsL1.define("bias", () => zeros([3]));
  // Check that they were set on the original params.
  assert(p.has("L1/weights"));
  assert(p.has("L1/bias"));
  assert(p.get("L1/weights") === w);
  assert(p.get("L1/bias") === b);
});
