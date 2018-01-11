const buildGpu = false;

if (process.env.PROPEL_BUILD_GPU && 
    Number(process.env.PROPEL_BUILD_GPU) !== 0) {
  if (process.platform !== "linux") {
    throw Error("PROPEL_BUILD_GPU only supported on linux");
  }
  buildGpu = true;
}

const platform = exports.platform = {
  "win32": "windows",
  "linux": "linux",
  "darwin": "mac",
}[process.platform] + (buildGpu ? "_gpu" : "");

exports.tfPkg = "propel_" + platform;
