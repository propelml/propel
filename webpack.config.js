var webpack = require('webpack');

module.exports = {
  entry: {
    'notebook': './notebook.ts',
    'propel': './api.ts',
    'propel.min': './api.ts',
    'test_isomorphic': './test_isomorphic.ts'
  },
  plugins: [
    new webpack.optimize.UglifyJsPlugin({
      minimize: true,
      sourceMap: true,
      include: /\.min\.js$/,
    })
  ],
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
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
