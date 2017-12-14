import * as mnist from "./mnist";
import { assert, assertAllEqual, assertShapesEqual } from "./util";

function testTrainSplit() {
  const dataset = mnist.load("train", 256);
  const [images, labels] = dataset.next();
  assertShapesEqual(images.shape, [256, 28, 28]);
  console.log("images dtype", images.dtype);
  // assert(images.dtype === "uint8");
  assertShapesEqual(labels.shape, [256]);
  // assert(labels.dtype === "uint8");
  assertAllEqual(labels.slice([0], [3]), [5, 0, 4]);
}

function testTestSplit() {
  const dataset = mnist.load("test", 16);
  const [images, labels] = dataset.next();
  assertShapesEqual(images.shape, [16, 28, 28]);
  // assert(images.dtype === "uint8");
  assertShapesEqual(labels.shape, [16]);
  // assert(labels.dtype === "uint8");
  assertAllEqual(labels.slice([0], [3]), [7, 2, 1]);
}

testTrainSplit();
testTestSplit();
