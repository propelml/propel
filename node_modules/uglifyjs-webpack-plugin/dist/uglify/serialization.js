'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});
/* eslint-disable no-new-func */
var toType = function toType(value) {
  return Object.prototype.toString.call(value).slice(8, -1);
};

var encode = exports.encode = function encode(key, value) {
  var type = toType(value);
  if (encode[type]) {
    return `<${type}>${encode[type](value, key)}`;
  }
  return value;
};

encode.RegExp = function (value) {
  return String(value);
};
encode.Function = function (value) {
  return String(value);
};

var decode = exports.decode = function decode(key, value) {
  if (typeof value === 'string') {
    var regex = /^<([A-Z]\w+)>([\w\W]*)$/;
    var match = value.match(regex);
    if (match && decode[match[1]]) {
      return decode[match[1]](match[2], key);
    }
  }

  return value;
};

decode.RegExp = function (value) {
  return Function(`return ${value}`)();
};
decode.Function = function (value, key) {
  return Function(`
  try {
    return ${value}.apply(null, arguments);
  } catch(err) {
    throw new Error('the option "${key}" performs an error in the child process: ' + err.message);
  }
`);
};