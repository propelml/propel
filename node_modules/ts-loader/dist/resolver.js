"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var node = require("enhanced-resolve/lib/node");
function makeResolver(options) {
    return node.create.sync(options.resolve);
}
exports.makeResolver = makeResolver;
