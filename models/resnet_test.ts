import { test } from "../tools/tester";
import * as pr from "../src/api";
import * as resnet from "./resnet";
import { assertShapesEqual } from "../src/tensor_util";
import { assert } from "../src/util";

test(async function resnet_resnet50() {
  const p = pr.params();
  const bs = 2;
  const x = pr.zeros([bs, 224, 224, 3]);
  const labels = pr.zeros([bs], { dtype: "int32" });
  const numClasses = 5;
  const logits = resnet.resnet50(x, p, numClasses);
  assertShapesEqual(logits.shape, [bs, numClasses]);
  const l = resnet.loss(logits, labels);
  assertShapesEqual(l.shape, []);
  assert(l.dataSync()[0] > 0, "Loss is positive");
});

test(async function resnet_simple() {
  const p = pr.params();
  const bs = 2;
  const x = pr.zeros([bs, 32, 32, 3]);
  const labels = pr.zeros([bs], { dtype: "int32" });
  const numClasses = 100;
  const logits = resnet.resnetSimple(x, p, 2, numClasses);
  assertShapesEqual(logits.shape, [bs, numClasses]);
  const l = resnet.loss(logits, labels);
  assertShapesEqual(l.shape, []);
  assert(l.dataSync()[0] > 0, "Loss is positive");
});

test(async function resnet_simpleGrad() {
  const bs = 2;
  const x = pr.zeros([bs, 32, 32, 3]);
  const numClasses = 10;
  const f = (params) => {
    const logits = resnet.resnetSimple(x, params, 2, numClasses);
    return logits.reduceSum();
  };
  const g = pr.gradParams(f);
  g(pr.params());
});
