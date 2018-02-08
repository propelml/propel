// This file is to avoid webpack's aggressive compilation.
// Note the noParse section of webpack.config.js
// It is also used as the main script of the modules to load the TensorFlow
// binding. See the package.json in propel_windows, propel_mac, propel_linux,
// and propel_linux_gpu.

module.exports = (function() {
  const toAttempt = [
    "./tensorflow-binding.node",
    "./build/Release/tensorflow-binding.node",
    "./build/Debug/tensorflow-binding.node",
    "../build/Release/tensorflow-binding.node",
    "../build/Debug/tensorflow-binding.node",
    "propel_linux_gpu",
    "propel_linux",
    "propel_mac",
    "propel_windows",
  ];

  for (const m of toAttempt) {
    try {
      return require(m);
    } catch(e) {
      // Ignore "module not found" errors.
      if (e.code !== "MODULE_NOT_FOUND")
        throw e;
    }
  }
  return null;
})();

