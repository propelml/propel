var webpack = require('webpack');

module.exports = {
  entry: {
    'notebook': './notebook.ts',
    'propel': './propel.ts',
    'propel.min': './propel.ts'
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
  output: {
    filename: '[name].js',
    path: __dirname + '/website/dist',
    libraryTarget: "var",
    library: "propel"
  }
};
