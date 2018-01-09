
const platform = exports.platform = {
  "win32": "windows",
  "linux": "linux",
  "darwin": "mac",
}[process.platform];

exports.tfPkg = "propel_" + platform;
