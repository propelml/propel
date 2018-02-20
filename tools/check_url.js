const { get } = require("https");
const url = process.argv[2];

get(url, ({ statusCode }) => {
  console.error(`HTTP status ${statusCode}: ${url}`);
  process.exit(statusCode === 200 ? 0 : 1);
});
