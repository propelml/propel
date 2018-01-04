#!/usr/bin/env node
const spawnSync = require('child_process').spawnSync;

process.chdir(__dirname + "/.."); // Go to project root.

function runParcel(fn) {
  const args = [
    './node_modules/.bin/parcel',
    'build',
    fn,
    '--out-dir',
    'website/dist/',
  ];
  console.log("Run", args.join(' '));
  const code = spawnSync(process.execPath, args, { stdio: 'inherit' });
  if (code !== 0) {
    console.log("Parcel completed with error.");
    process.exit(1);
  }
}

runParcel("nn_example.ts");
runParcel("notebook.ts");
runParcel("test_isomorphic.ts");
