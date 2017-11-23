
let tf;
try {
  tf = require('./build/Debug/tensorflow-binding.node');
} catch (e) {
  tf = require('./build/Release/tensorflow-binding.node');
}
let ctx = new tf.Context();

let typedArray = new Uint16Array([1, 2, 3, 4, 5, 6]);
let tensor = new tf.Tensor(typedArray, [2, 3]);
console.assert(tensor.device == "CPU:0");

console.log("PASS");
