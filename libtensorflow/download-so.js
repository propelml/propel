
/* Download libtensorflow binaries mirrored from the TensorFlow CI.
 * Original URLs:
 * http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow/TYPE=mac-slave/lastSuccessfulBuild/artifact/lib_package/libtensorflow-cpu-darwin-x86_64.tar.gz
 * http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow/TYPE=cpu-slave/lastSuccessfulBuild/artifact/lib_package/libtensorflow-cpu-linux-x86_64.tar.gz
 */
const downloadUrl = {
  darwin: 'http://propelml.org/libtensorflow_20171121/libtensorflow-cpu-darwin-x86_64.tar.gz',
  linux: 'http://propelml.org/libtensorflow_20171121/libtensorflow-cpu-linux-x86_64.tar.gz'
}[process.platform];

const outDir = process.argv[2] || '';

const fs = require('fs');
const http = require('http');
const path = require('path');
const tar = require('tar');

fetch(downloadUrl);

function fetch(url) {
  console.error('Downloading %s', url);

  http.get(url, (res) => {
    if (res.statusCode === 502)
      return fetch(url); // Tensorflow CI server can be very flaky at times.
    else if (res.statusCode !== 200)
      throw new Error("Download failed: HTTP " + res.statusCode + "\n" + url);

    res.pipe(new tar.Parse({
      onentry(entry) {
        let name = path.basename(entry.header.path);
        let ext = path.extname(name);

        if (ext === '.so') {
          console.error('Extracting %s', name);
          let outPath = path.resolve(outDir, name);
          entry.pipe(fs.createWriteStream(outPath));
        }

        entry.resume();
      },

      onwarn: console.warn
    }));
  });
}
