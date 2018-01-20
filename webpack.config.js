const webpack = require("webpack");
const config = require("./tools/config");

const common = {
  plugins: [ ],
  module: {
    noParse: /load_binding/,
    rules: [
      {
        test: /\.ts$/,
        loader: 'ts-loader',
        options: {
          compilerOptions: {
            "declaration": false,
            "target": "es5",
            sourceMap: true
          }
        }
      }
    ]
  },
  resolve: {
    extensions: [ '.ts', '.js' ]
  },
  devtool: "source-map"
};

function webConfig(name, libraryName) {
  const cfg = Object.assign({}, common);
  cfg.target = "web";
  cfg.name = name;
  cfg.plugins.push(new webpack.DefinePlugin({ IS_WEB: true }));
  cfg.node = { "fs": "empty" };
  cfg.output = {
    filename: name + ".js",
    path: __dirname + "/build/propel/",
    library: libraryName,
    libraryTarget: 'var'
  };
  return cfg;
}

function nodeConfig(name) {
  const cfg = Object.assign({}, common);
  cfg.target = "node";
  cfg.name = name;
  cfg.plugins.push(new webpack.DefinePlugin({ IS_WEB: false }));
  cfg.node = {
    "fs": false,
  };
  cfg.output = {
    filename: name + ".js",
    path: __dirname + "/build/propel/",
    library: '',
    libraryTarget: 'commonjs'
  };
  return cfg;
}

const propelWeb = webConfig("propel_web", "propel");
propelWeb.entry = "./api.ts";

const propelNode = nodeConfig("propel_node");
propelNode.entry = "./api.ts";

const testsIso = webConfig("test_isomorphic", "yyy");
testsIso.output.path = `${__dirname}/build/website`;
testsIso.entry = "./test_isomorphic.ts";

const testsDl = webConfig("test_dl", "zzz");
testsDl.output.path = `${__dirname}/build/website`;
testsDl.entry = "./test_dl.ts";

const website = webConfig("website", "vvv");
website.output.path = `${__dirname}/build/website`;
website.entry = "./website.ts";

/*
const testsNode = nodeConfig("tests_node");
testsNode.entry = "./test_isomorphic.ts";

const tfNode = nodeConfig(config.tfPkg);
tfNode.output.filename = "tf.js";
tfNode.entry = "./tf.ts";
*/

module.exports = [
  propelNode,
  propelWeb,
  testsIso,
  testsDl,
  website,
];
