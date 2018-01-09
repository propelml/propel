var webpack = require('webpack');

module.exports = {
  entry: {
    'nn_example': './nn_example.ts',
    'notebook': './notebook.ts',
    'propel': './api.ts',
    'test_dl': './test_dl.ts',
    'test_isomorphic': './test_isomorphic.ts'
  },
  plugins: [
    new webpack.optimize.UglifyJsPlugin({
      minimize: true,
      sourceMap: true,
      include: /\.min\.js$/,
    }),
    new webpack.LoaderOptionsPlugin({
      debug: true
    }),
  ],
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: {
          loader: 'ts-loader',
          options: {
            compilerOptions: {
              "target": "es5"
            }
          }
        }
      }
    ]
  },
  devtool: 'source-map',
  resolve: {
    extensions: [ '.ts', '.js' ]
  },
  node: { fs: "empty" },
  output: {
    filename: '[name].js',
    path: __dirname + '/website/dist',
    libraryTarget: "var",
    library: "propel"
  }
};
