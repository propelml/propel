const { execSync, spawnSync } = require("child_process");
const fs = require("fs");
const { resolve } = require("path");
const rimraf = require("rimraf");

// Be extra careful to enable ts-node type checking.
process.env.TS_NODE_TYPE_CHECK = true;

require("ts-node").register({"typeCheck": true });
const gendoc = require("./gendoc.ts");

// Always chdir to propel's project root.
const root = resolve(__dirname, "..");
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

function symlink(a, b) {
  let existing;
  try {
    existing = fs.readlinkSync(b);
  } catch (e) {
    if (e.code !== "ENOENT") throw e;
  }

  if (existing) {
    if (resolve(b, a) === resolve(b, existing)) {
      console.log("symlink already exists", a, b);
      return;
    } else {
      // Remove the existing symlink.
      fs.unlinkSync(b);
    }
  }

  console.log("symlink", a, b);

  if (process.platform === "win32") {
    // Until https://github.com/libuv/libuv/pull/1706 makes it into a node,
    // can't create symlinks to directories on Windows.
    execSync(`mklink /d "${b}" "${a}"`);
  } else {
    fs.symlinkSync(a, b, "dir");
  }
}

function tsnode(args, env = {}) {
  sh("node ./node_modules/ts-node/dist/bin.js --type-check " + args, env);
}

const parcelCli = "./node_modules/parcel-bundler/bin/cli.js";

function parcel(inFile, outDir, debug) {
  debug = debug != null ? !!debug : process.argv.indexOf("debug") >= 1;
  const Bundler = require("parcel-bundler");
  const bundler = new Bundler(inFile, {
    cache: true,
    logLevel: process.env.CI ? 1 : null,
    minify: !debug,
    outDir,
    production: !debug,
    watch: false
  });
  return bundler.bundle(); // Returns Promise.
}

function version() {
  let pkg = JSON.parse(fs.readFileSync("package.json", "utf8"));
  return pkg.version;
}

exports.gendoc = (fn) => {
  let gendocFlag = (process.argv.indexOf("gendoc") >= 0);
  if (gendocFlag || !fs.existsSync(fn)) {
    const docs = gendoc.genJSON();
    const docsJson = JSON.stringify(docs, null, 2);
    fs.writeFileSync(fn, docsJson);
  }
}


exports.mkdir = mkdir;
exports.parcel = parcel;
exports.parcelCli = parcelCli;
exports.rmrf = rmrf;
exports.root = root;
exports.sh = sh;
exports.symlink = symlink;
exports.tsnode = tsnode;
exports.version = version;
