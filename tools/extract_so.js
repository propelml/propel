// Extract libtensorflow binaries mirrored from the TensorFlow CI.
// Original URLs:
// tslint:disable-next-line:max-line-length
// http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow/TYPE=mac-slave/lastSuccessfulBuild/artifact/lib_package/libtensorflow-cpu-darwin-x86_64.tar.gz
// tslint:disable-next-line:max-line-length
// http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow/TYPE=cpu-slave/lastSuccessfulBuild/artifact/lib_package/libtensorflow-cpu-linux-x86_64.tar.gz

const hasCuda = (require('child_process').spawnSync('nvidia-smi').status === 0);
const platform = process.platform + '-' + (hasCuda ? 'gpu' : 'cpu');

const tarFn = {
  'darwin-cpu': 'libtensorflow-cpu-darwin-x86_64.tar.gz',
  'linux-cpu': 'libtensorflow-cpu-linux-x86_64.tar.gz',
  'linux-gpu': 'libtensorflow-gpu-linux-x86_64.tar.gz',
}[platform];

const outDir = process.argv[2] || '';
console.error('outDir %s', outDir);

const fs = require('fs');
const http = require('http');
const path = require('path');
const tar = require('tar');

let tarPath = path.join(__dirname, '..', 'deps', 'libtensorflow', tarFn);
console.error('Extracting %s', tarPath);

tar.t({
  file: tarPath,
  onwarn: console.warn,
  onentry(entry) {
    let name = path.basename(entry.header.path);
    let ext = path.extname(name);

    if (ext === '.so') {
      console.error('Extracting %s', name);
      let outPath = path.resolve(outDir, name);
      console.log(outPath)
      entry.pipe(fs.createWriteStream(outPath));
    }
  }
})
