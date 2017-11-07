'use strict';

var _minify = require('./minify');

var _minify2 = _interopRequireDefault(_minify);

var _serialization = require('./serialization');

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

module.exports = function (options, callback) {
  try {
    callback(null, (0, _minify2.default)(JSON.parse(options, _serialization.decode)));
  } catch (errors) {
    callback(errors);
  }
};