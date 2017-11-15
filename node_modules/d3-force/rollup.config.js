const definition = require("./package.json");
const dependencies = Object.keys(definition.dependencies);

export default {
  input: "index",
  external: dependencies,
  output: {
    extend: true,
    file: `build/${definition.name}.js`,
    format: "umd",
    globals: dependencies.reduce((p, v) => (p[v] = "d3", p), {}),
    name: "d3"
  }
};
