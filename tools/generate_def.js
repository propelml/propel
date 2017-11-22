
const fs = require('fs');

const files = process.argv.slice(2);

const symbols = files
  .map((file) => fs.readFileSync(file))
  .join('\n')
  .split('\n')
  .map((line) => {
    var match = /^TF_CAPI_EXPORT.*?\s+(\w+)\s*\(/.exec(line);
    return match && match[1];
  })
  .filter((symbol) => symbol !== null);

process.stdout.write('EXPORTS\n' + symbols.join('\n'));