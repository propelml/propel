export default {
  input: "index",
  external: [
    "d3-collection",
    "d3-dispatch",
    "d3-dsv"
  ],
  output: {
    extend: true,
    file: "build/d3-request.js",
    format: "umd",
    globals: {
      "d3-collection": "d3",
      "d3-dispatch": "d3",
      "d3-dsv": "d3"
    },
    name: "d3"
  }
};
