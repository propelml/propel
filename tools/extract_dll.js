// Extract libtensorflow binaries mirrored from the TensorFlow CI.
// Original URL:
// tslint:disable-next-line:max-line-length
// http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow-windows/lastStableBuild/artifact/lib_package/libtensorflow-cpu-windows-x86_64.zip

// Work around a bug where node-gyp on windows adds a trailing " to PRODUCT_DIR.
let outDir = process.argv[2] || '';
outDir = outDir.replace(/"$/, '');

const fs = require('fs');
const http = require('http');
const path = require('path');
const yauzl = require('yauzl');

let zipFilename = path.resolve(__dirname, '../deps/libtensorflow/' +
  'libtensorflow-cpu-windows-x86_64.zip')

yauzl.open(zipFilename, (err, zip) => {
  if (err)
    throw err;

  zip.on('entry', (entry) => {
    let name = path.basename(entry.fileName);
    let ext = path.extname(name);

    if (ext !== '.dll')
      return;

    console.error('Extracting %s', name);

    zip.openReadStream(entry, (err, stream) => {
      if (err)
        throw err;

      let outPath = path.resolve(outDir, name);
      console.log(outPath)
      stream.pipe(fs.createWriteStream(outPath));
    });
  });
});
