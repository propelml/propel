
// The original Residual Network (ResNet) image classifier by
//
//   Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
//   Deep Residual Learning for Image Recognition. arXiv:1512.03385
//
// Implementation based on Slim's resnet_v1.py by Nathan Silberman
// Sergio Guadarrama and others at Google Brain.

import {
  Params,
  Tensor,
  tensor,
} from "../src/api";
import { assertShapesEqual } from "../src/tensor_util";
import { assert } from "../src/util";

const weightDecay = 0.0001

export function loss(params: Params, logits: Tensor, labels: Tensor, numClasses= 1000) {
  const softmaxLoss = logits.softmaxLoss(labels);
  return softmaxLoss.add(weightDecayLoss(params));
}

function weightDecayLoss(params: Params): Tensor {
  let loss = tensor(0);
  for (let [name, p] of params) {
    if (name.endsWith("filter") || name.endsWith("weights")) {
      let l2norm = p.square().reduceSum();
      loss = loss.add(l2norm);
    }
  }
  return loss.mul(weightDecay);
}

/** ResNet used in sect 4.2 of https://arxiv.org/abs/1512.03385
 * Used for CIFAR10.
 * The network inputs are 32×32 images, with the per-pixel mean subtracted.
 */
export function resnetSimple(
  images: Tensor,
  params: Params,
  n = 18,  // 110 layer network. 6n+2 = number of layers.
  numClasses = 10,
): Tensor {
  let x = images;
  x = conv2dSame(x, params.scope("root"), 16, 3, 1);

  x = stackBlocks(x, params, {
    blocks: [
      ["block1", 16, n, 2],
      ["block2", 32, n, 2],
      ["block3", 64, n, 1],
    ],
    numClasses,
    unitFn: basicUnit,
  });

  x = x.reduceMean([1, 2], true); // avgPool
  return x.linear("FC", params, numClasses);
}

/** ResNet with 18 convolutions. Returns logits.
 * Used to train cifar10.
 */
export function resnet50(images: Tensor, params: Params,
                         numClasses = 1000): Tensor {
  return resnet(images, params.scope("resnet50"), {
    blocks: [
      ["block1", 64, 3, 2],
      ["block2", 128, 4, 2],
      ["block3", 256, 6, 2],
      ["block4", 512, 3, 1],
    ],
    numClasses,
    unitFn: bottleneckUnit,
  });
}

// scope, baseDepth, numUnits, stride
type BlockDesc = [string, number, number, number];

interface ResNetOpts {
  blocks: BlockDesc[];
  numClasses: number;
  unitFn(x: Tensor, params: Params, args: LayerArgs): Tensor;
}

interface LayerArgs {
  baseDepth: number;
  stride: number;
}

// Training for ImageNet uses [224, 224, 3] inputs and produces
// [7, 7, 1000] feature maps.
export function resnet(
  input: Tensor,
  params: Params,
  opts: ResNetOpts
): Tensor {
  const bs = input.shape[0];
  const isDefaultShape = input.shape.length === 4 &&
                         input.shape[1] === 224 && input.shape[2] === 224;
  let x = input;

  // Root layer.
  x = conv2dSame(x, params.scope("root"), 64, 7, 2);
  if (isDefaultShape) {
    assertShapesEqual(x.shape, [bs, 112, 112, 64 ]);
  }

  x = x.maxPool({ size: 3, stride: 2, padding: "same" });
  if (isDefaultShape) {
    assertShapesEqual(x.shape, [bs, 56, 56, 64]);
  }

  x = stackBlocks(x, params, opts); // Main network.
  if (isDefaultShape) {
    assert(x.shape[1] === 7 && x.shape[2] === 7);
  }

  x = x.reduceMean([1, 2], true); // avgPool
  return x.linear("FC", params, opts.numClasses);
}

function stackBlocks(x: Tensor, params: Params, opts: ResNetOpts): Tensor {
  for (let i = 0; i < opts.blocks.length; i++) {
    const [name, baseDepth, numUnits, stride] = opts.blocks[i];
    const blockParams = params.scope(name);
    for (let j = 0; j < numUnits; j++) {
      const isLast = (j === numUnits - 1);
      const unitParams = blockParams.scope("unit" + j);
      x = opts.unitFn(x, unitParams, {
        baseDepth,
        stride: isLast ? stride : 1,
      });
    }
  }
  return x;
}

function subsample(x: Tensor, factor: number) {
  if (factor === 1) {
    return x;
  } else {
    return x.maxPool({size: 1, stride: factor, padding: "same" });
  }
}

// Depicted in figure 5, right.
export function bottleneckUnit(
  x: Tensor,
  params: Params,
  {stride, baseDepth}: LayerArgs
): Tensor {
  const depth = 4 * baseDepth;
  const depthBottleneck = baseDepth;
  const depthIn = x.shape[3];

  let shortcut;

  if (depth === depthIn) {
    shortcut = subsample(x, stride);
  } else {
    shortcut = x.conv2d("shortcut", params, depth, { size: 1, stride })
                .batchNorm("shortcut", params);
  }

  let residual = x
    .conv2d("conv1", params, depthBottleneck, { size: 1, stride: 1 })
    .batchNorm("conv1", params)
    .relu();
  // conv2 should be dilated here?
  residual = residual
    .conv2d("conv2", params, depthBottleneck, {
      padding: "same",
      size: 3,
      stride,
    })
    .batchNorm("conv2", params)
    .relu();
  residual = residual
    .conv2d("conv3", params, depth, { size: 1, stride: 1 })
    .batchNorm("conv3", params);

  assertShapesEqual(shortcut.shape, residual.shape);
  return shortcut.add(residual).relu();
}

// Depicted in figure 5, left.
// Non-bottleneck unit.
export function basicUnit(
  x: Tensor,
  params: Params,
  {stride, baseDepth}: LayerArgs
): Tensor {
  const depthIn = x.shape[3];
  let shortcut;

  if (baseDepth === depthIn) {
    shortcut = subsample(x, stride);
  } else {
    shortcut = x.conv2d("shortcut", params, baseDepth, { size: 1, stride })
                .batchNorm("shortcut", params);
  }

  let residual = x
    .conv2d("conv1", params, baseDepth, { size: 3, stride: 1 })
    .batchNorm("bn1", params)
    .relu();
  residual = residual.conv2d("conv2", params, baseDepth, { size: 3, stride });
  residual = residual.batchNorm("bn2", params);

  assertShapesEqual(shortcut.shape, residual.shape);
  return shortcut.add(residual).relu();
}

// Port of tensorflow/contrib/slim/nets/resnet_utils.py conv2d_same
// When stride > 1, then we do explicit zero-padding, followed by conv2d with
// 'VALID' padding.
function conv2dSame(inputs, params, numOutputs, kernelSize, stride) {
  const rate = 1;  // Dilated convolutions not supported.
  let x = inputs;
  if (stride === 1) {
    x = x.conv2d("conv", params, numOutputs, {
      bias: false,
      padding: "same",
      size: kernelSize,
      stride,
    });
  } else {
    const kernelSizeEffective = kernelSize + (kernelSize - 1) * (rate - 1);
    const padTotal = kernelSizeEffective - 1;
    const padBeg = Math.floor(padTotal / 2);
    const padEnd = padTotal - padBeg;
    x = x.pad([[0, 0], [padBeg, padEnd], [padBeg, padEnd], [0, 0]]);
    x = x.conv2d("conv", params, numOutputs, {
      bias: false,
      padding: "valid",
      size: kernelSize,
      stride,
    });
  }
  return x.batchNorm("bn", params).relu();
}
