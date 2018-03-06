// These test cases are adapted from TensorFlow:
// tslint:disable-next-line:max-line-length
// https://github.com/tensorflow/tensorflow/blob/f47b6c9ec5e6c4561a6ed97ef2342ea737dcd80c/tensorflow/python/kernel_tests/conv_ops_test.py#L348-L863

export interface ConvTestCase {
  name: string;
  expected: number[];
  inputShape: [number, number, number, number]; // NHWC
  filterShape: [number, number, number, number]; // H W InChans OutChans
  outputShape?: [number, number, number, number];
  strides: [number, number]; // [column, row ]
  padding: "same" | "valid";
  err?: number;
  skip?: string;  // tf or dl to skip certain backends.
}

const fw: ConvTestCase[] = [
  {
    name: "Conv2D1x1Filter",
    expected: [
      30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0,
      171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0
    ],
    inputShape: [1, 2, 3, 3],
    filterShape: [1, 1, 3, 3],
    strides: [1, 1],
    padding: "valid",
  },
  {
    name: "Conv2DEmpty",
    expected: [],
    inputShape: [0, 2, 3, 3],
    filterShape: [1, 1, 3, 3],
    strides: [1, 1],
    padding: "valid",
    skip: "dl",  // FIXME
  },
  {
    name: "Conv2D2x2Filter",
    expected: [2271.0, 2367.0, 2463.0, 2901.0, 3033.0, 3165.0],
    inputShape: [1, 2, 3, 3],
    filterShape: [2, 2, 3, 3],
    strides: [1, 1],
    padding: "valid",
  },
  {
    name: "Conv2D1x2Filter",
    expected: [
      231.0, 252.0, 273.0, 384.0, 423.0, 462.0, 690.0, 765.0, 840.0, 843.0,
      936.0, 1029.0
    ],
    inputShape: [1, 2, 3, 3],
    filterShape: [1, 2, 3, 3],
    strides: [1, 1],
    padding: "valid",
  },
  {
    name: "Conv2D2x2FilterStride2",
    expected: [2271.0, 2367.0, 2463.0],
    inputShape: [1, 2, 3, 3],
    filterShape: [2, 2, 3, 3],
    strides: [2, 2],
    padding: "valid",
  },
  {
    name: "Conv2D2x2FilterStride2Same",
    expected: [2271.0, 2367.0, 2463.0, 1230.0, 1305.0, 1380.0],
    inputShape: [1, 2, 3, 3],
    filterShape: [2, 2, 3, 3],
    strides: [2, 2],
    padding: "same",
  },
  {
    name: "Conv2D2x2FilterStride1x2",
    expected: [58.0, 78.0, 98.0, 118.0, 138.0, 158.0],
    inputShape: [1, 3, 6, 1],
    filterShape: [2, 2, 1, 1],
    strides: [1, 2],
    padding: "valid",
  },
  {
    name: "Conv2DKernelSmallerThanStrideValid",
    expected: [65, 95, 275, 305],
    inputShape: [1, 7, 7, 1],
    filterShape: [2, 2, 1, 1],
    strides: [3, 3],
    padding: "valid",
  },
  {
    name: "Conv2DKernelSmallerThanStrideSame1",
    expected: [1, 3, 7, 9],
    inputShape: [1, 3, 3, 1],
    filterShape: [1, 1, 1, 1],
    strides: [2, 2],
    padding: "same",
  },
  {
    name: "Conv2DKernelSmallerThanStrideSame2",
    expected: [1, 3, 9, 11],
    inputShape: [1, 4, 4, 1],
    filterShape: [1, 1, 1, 1],
    strides: [2, 2],
    padding: "same",
    skip: "dl",  // FIXME
  },
  {
    name: "Conv2DKernelSmallerThanStrideSame3",
    expected: [44, 28, 41, 16],
    inputShape: [1, 4, 4, 1],
    filterShape: [2, 2, 1, 1],
    strides: [3, 3],
    padding: "same",
  },
  {
    name: "Conv2DKernelSizeMatchesInputSize",
    expected: [50, 60],
    inputShape: [1, 2, 2, 1],
    filterShape: [2, 2, 1, 2],
    strides: [1, 1],
    padding: "valid",
  },
];

const bwInput: ConvTestCase[] = [
  {
    name: "Conv2D2x2Depth1ValidBackpropInput",
    expected: [1.0, 4.0, 4.0, 3.0, 10.0, 8.0],
    inputShape: [1, 2, 3, 1],
    filterShape: [2, 2, 1, 1],
    outputShape: [1, 1, 2, 1],
    strides: [1, 1],
    padding: "valid",
    err: 1e-5
  },
  {
    name: "Conv2DEmptyBackpropInput",
    expected: [],
    inputShape: [0, 2, 3, 1],
    filterShape: [2, 2, 1, 1],
    outputShape: [0, 1, 2, 1],
    strides: [1, 1],
    padding: "valid",
    err: 1e-5,
    skip: "dl",
  },
  {
    name: "Conv2D2x2Depth3ValidBackpropInput",
    expected: [
      14.0, 32.0, 50.0, 100.0, 163.0, 226.0, 167.0, 212.0, 257.0, 122.0,
      140.0, 158.0, 478.0, 541.0, 604.0, 437.0, 482.0, 527.0
    ],
    inputShape: [1, 2, 3, 3],
    filterShape: [2, 2, 3, 3],
    outputShape: [1, 1, 2, 3],
    strides: [1, 1],
    padding: "valid",
    err: 1e-4
  },
  {
    name: "Conv2D2x2Depth3ValidBackpropInputStride1x2",
    expected: [
      1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 7.0, 12.0, 11.0, 18.0, 15.0, 24.0, 12.0,
      16.0, 15.0, 20.0, 18.0, 24.0
    ],
    inputShape: [1, 3, 6, 1],
    filterShape: [2, 2, 1, 1],
    outputShape: [1, 2, 3, 1],
    strides: [1, 2],
    padding: "valid",
    err: 1e-5
  },
  {
    name: "Conv2DStrideTwoFilterOneSameBackpropInput",
    expected: [
      1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0,
      0.0, 0.0
    ],
    inputShape: [1, 4, 4, 1],
    filterShape: [1, 1, 1, 1],
    outputShape: [1, 2, 2, 1],
    strides: [2, 2],
    padding: "same",
    err: 1e-5,
    skip: "dl",
  },
  {
    name: "Conv2DKernelSizeMatchesInputSizeBackpropInput",
    expected: [5.0, 11.0, 17.0, 23.0],
    inputShape: [1, 2, 2, 1],
    filterShape: [2, 2, 1, 2],
    outputShape: [1, 1, 1, 2],
    strides: [1, 1],
    padding: "valid",
    err: 1e-5
  },
];

const bwFilter: ConvTestCase[] = [
  {
    name: "Conv2D2x2Depth1ValidBackpropFilter",
    expected: [5.0, 8.0, 14.0, 17.0],
    inputShape: [1, 2, 3, 1],
    filterShape: [2, 2, 1, 1],
    outputShape: [1, 1, 2, 1],
    strides: [1, 1],
    padding: "valid",
  },
  {
    name: "Conv2DEmptyBackpropFilter",
    expected: [],
    inputShape: [1, 2, 3, 1],
    filterShape: [2, 2, 1, 0],
    outputShape: [1, 1, 2, 0],
    strides: [1, 1],
    padding: "valid",
    skip: "dl",
  },
  {
    name: "Conv2D2x2Depth3ValidBackpropFilter",
    expected: [
      17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0, 32.0, 43.0, 54.0,
      37.0, 50.0, 63.0, 42.0, 57.0, 72.0, 62.0, 85.0, 108.0, 67.0, 92.0,
      117.0, 72.0, 99.0, 126.0, 77.0, 106.0, 135.0, 82.0, 113.0, 144.0, 87.0,
      120.0, 153.0
    ],
    inputShape: [1, 2, 3, 3],
    filterShape: [2, 2, 3, 3],
    outputShape: [1, 1, 2, 3],
    strides: [1, 1],
    padding: "valid",
  },
  {
    name: "Conv2D2x2Depth3ValidBackpropFilterStride1x2",
    expected: [161.0, 182.0, 287.0, 308.0],
    inputShape: [1, 3, 6, 1],
    filterShape: [2, 2, 1, 1],
    outputShape: [1, 2, 3, 1],
    strides: [1, 2],
    padding: "valid",
  },
  {
    name: "Conv2DStrideTwoFilterOneSameBackpropFilter",
    expected: [78.],
    inputShape: [1, 4, 4, 1],
    filterShape: [1, 1, 1, 1],
    outputShape: [1, 2, 2, 1],
    strides: [2, 2],
    padding: "same",
    skip: "dl",
  },
  {
    name: "Conv2DKernelSizeMatchesInputSizeBackpropFilter",
    expected: [1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0],
    inputShape: [1, 2, 2, 1],
    filterShape: [2, 2, 1, 2],
    outputShape: [1, 1, 1, 2],
    strides: [1, 1],
    padding: "valid",
  },
];

export let cases = {
  "bwFilter": bwFilter,
  "bwInput": bwInput,
  "fw": fw,
};
