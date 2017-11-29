let binding;
try {
  binding = require('./build/Debug/tensorflow-binding.node');
} catch (e) {
  binding = require('./build/Release/tensorflow-binding.node');
}
let ctx = new binding.Context();
console.assert(ctx instanceof binding.Context);

let typedArray = new Float32Array([1, 2, 3, 4, 5, 6]);
let a = new binding.Tensor(typedArray, [2, 3]);
let result = new Float32Array(a.asArrayBuffer());
console.assert(result.length == 6);
console.assert(result[0] == 1)
console.assert(result[1] == 2)

console.log("OK");
