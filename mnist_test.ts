import * as mnist from "./mnist";
import { assert, assertAllEqual, assertShapesEqual } from "./util";

function testTrainSplit() {
  const dataset = mnist.load("train", 256);
  dataset.next().then(({images, labels}) => {
    assertShapesEqual(images.shape, [256, 28, 28]);
    console.log("images dtype", images.dtype);
    // assert(images.dtype === "uint8");
    assertShapesEqual(labels.shape, [256]);
    // assert(labels.dtype === "uint8");
    // > hexdump deps/mnist/train-labels-idx1-ubyte | head -1
    // 0000000 00 00 08 01 00 00 ea 60 05 00 04 01 09 02 01 03
    //         ---magic--- ---items--- ---------labels--------
    assertAllEqual(labels.slice([0], [8]), [5, 0, 4, 1, 9, 2, 1, 3]);
    console.log("TrainSplit ok");
  });
}

function testTestSplit() {
  const dataset = mnist.load("test", 16);
  dataset.next().then(({images, labels}) => {
    assertShapesEqual(images.shape, [16, 28, 28]);
    // assert(images.dtype === "uint8");
    assertShapesEqual(labels.shape, [16]);
    // assert(labels.dtype === "uint8");
    assertAllEqual(labels.slice([0], [3]), [7, 2, 1]);
    console.log("TestSplit ok");
  });
}

testTrainSplit();
testTestSplit();
