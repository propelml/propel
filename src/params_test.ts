import { test } from "../tools/tester";
import { Params } from "./params";

test(async function params_smoke() {
  const params = new Params();
  // randn
  const a = params.randn("a", [2, 3]);
  assert(a.getData()[0] !== 0);
  assertShapesEqual(a.shape, [2, 3]);
  const aa = params.randn("a", [2, 3]);
  assertAllEqual(a, aa);
  // zeros
  const b = params.zeros("b", [2, 3]);
  assertAllEqual(b, [[0, 0, 0], [0, 0, 0]]);
  const bb = params.zeros("b", [2, 3]);
  assertAllEqual(b, bb);
});
