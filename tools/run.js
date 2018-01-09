const spawnSync = require("child_process").spawnSync;
const fs = require("fs");
const root = require("path").dirname(__dirname);
const rimraf = require("rimraf");
// Always chdir to propel's project root.
process.chdir(root);

/** Runs a new subprocess synchronously.
  * We use this instead of shell scripts to support Windows.
  * The process arguments are forwarded to commands.
  * This is so one can run ./tools/tslint.js --fix
  */
function sh(cmd, env = {}) {
  let args = cmd.split(/\s+/);
  console.log(args.join(" "));
  let exe = args.shift();
  // Use this node if node is specified.
  if (exe === "node") exe = process.execPath;
  let r = spawnSync(exe, args, {
    stdio: 'inherit',
    env: { ...process.env, ...env }
  });
  if (r.error) throw r.error;
  if (r.status) {
    console.log("Error", args[0]);
    process.exit(r.status);
  }
}

function mkdir(p) {
  if (!fs.existsSync(p)) {
    console.log("mkdir", p);
    fs.mkdirSync(p);
  }
}

function rmrf(d) {
  if (fs.existsSync(d)) {
    console.log("Delete dir", d);
    rimraf.sync(d);
  }
}

function version() {
  let pkg = JSON.parse(fs.readFileSync("package.json", "utf8"));
  return pkg.version;
}

exports.sh = sh;
exports.mkdir = mkdir;
exports.rmrf = rmrf;
exports.root = root;
exports.version = version;
