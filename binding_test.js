let binding;
try {
  binding = require('./build/Debug/tensorflow-binding.node');
} catch (e) {
  binding = require('./build/Release/tensorflow-binding.node');
}
let ctx = new binding.Context();
console.assert(ctx instanceof binding.Context);

let typedArray = new Uint16Array([1, 2, 3, 4, 5, 6]);
let tensor = new binding.Tensor(typedArray, [2, 3]);
console.assert(tensor.device == "CPU:0");

console.log("PASS");
