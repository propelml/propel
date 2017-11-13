var webpack = require('webpack');

module.exports = {
  entry: {
    'repl': './repl.ts',
    'sigprop': './sigprop.ts',
    'sigprop.min': './sigprop.ts'
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
    path: __dirname + '/dist',
    libraryTarget: "var",
    library: "sigprop"
  }
};
