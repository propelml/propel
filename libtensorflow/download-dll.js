
/* Download libtensorflow binaries mirrored from the TensorFlow CI.
 * Original URL:
 * http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow-windows/lastStableBuild/artifact/lib_package/libtensorflow-cpu-windows-x86_64.zip
 */
let downloadUrl = 'http://propelml.org/libtensorflow_20171121/libtensorflow-cpu-windows-x86_64.zip';

// Work around a bug where node-gyp on windows adds a trailing " to PRODUCT_DIR.
let outDir = process.argv[2] || '';
outDir = outDir.replace(/"$/, '');

const fs = require('fs');
const http = require('http');
const path = require('path');
const yauzl = require('yauzl');

fetch(downloadUrl);

function fetch(url) {
  console.error('Downloading %s', url);

  http.get(url, (res) => {
    if (res.statusCode === 502)
      return fetch(url); // Tensorflow CI server can be very flaky at times.
    else if (res.statusCode !== 200)
      throw new Error("Download failed: HTTP " + res.statusCode + "\n" + url);

    let buffers = [];
    res.on('data', (buf) => buffers.push(buf));

    res.on('end', () => {
      yauzl.fromBuffer(Buffer.concat(buffers), (err, zip) => {
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
            stream.pipe(fs.createWriteStream(outPath));
          });
        });
      });
    });
  });
}
