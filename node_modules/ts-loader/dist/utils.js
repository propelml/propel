"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var path = require("path");
var fs = require("fs");
var constants = require("./constants");
function registerWebpackErrors(existingErrors, errorsToPush) {
    Array.prototype.splice.apply(existingErrors, [0, 0].concat(errorsToPush));
}
exports.registerWebpackErrors = registerWebpackErrors;
function hasOwnProperty(obj, property) {
    return Object.prototype.hasOwnProperty.call(obj, property);
}
exports.hasOwnProperty = hasOwnProperty;
/**
 * The default error formatter.
 */
function defaultErrorFormatter(error, colors) {
    var messageColor = error.severity === 'warning' ? colors.bold.yellow : colors.bold.red;
    return colors.grey('[tsl] ') + messageColor(error.severity.toUpperCase()) +
        (error.file === ''
            ? ''
            : messageColor(' in ') + colors.bold.cyan(error.file + "(" + error.line + "," + error.character + ")")) + constants.EOL +
        messageColor("      TS" + error.code + ": " + error.content);
}
/**
 * Take TypeScript errors, parse them and format to webpack errors
 * Optionally adds a file name
 */
function formatErrors(diagnostics, loaderOptions, colors, compiler, merge) {
    return diagnostics
        ? diagnostics
            .filter(function (diagnostic) { return loaderOptions.ignoreDiagnostics.indexOf(diagnostic.code) === -1; })
            .map(function (diagnostic) {
            var file = diagnostic.file;
            var position = file === undefined ? undefined : file.getLineAndCharacterOfPosition(diagnostic.start);
            var errorInfo = {
                code: diagnostic.code,
                severity: compiler.DiagnosticCategory[diagnostic.category].toLowerCase(),
                content: compiler.flattenDiagnosticMessageText(diagnostic.messageText, constants.EOL),
                file: file === undefined ? '' : path.normalize(file.fileName),
                line: position === undefined ? 0 : position.line + 1,
                character: position === undefined ? 0 : position.character + 1
            };
            var message = loaderOptions.errorFormatter === undefined
                ? defaultErrorFormatter(errorInfo, colors)
                : loaderOptions.errorFormatter(errorInfo, colors);
            var error = makeError(message, merge === undefined ? undefined : merge.file, position === undefined
                ? undefined
                : { line: errorInfo.line, character: errorInfo.character });
            return Object.assign(error, merge);
        })
        : [];
}
exports.formatErrors = formatErrors;
function readFile(fileName, encoding) {
    if (encoding === void 0) { encoding = 'utf8'; }
    fileName = path.normalize(fileName);
    try {
        return fs.readFileSync(fileName, encoding);
    }
    catch (e) {
        return undefined;
    }
}
exports.readFile = readFile;
function makeError(message, file, location) {
    return {
        message: message, location: location, file: file,
        loaderSource: 'ts-loader'
    };
}
exports.makeError = makeError;
function appendSuffixIfMatch(patterns, path, suffix) {
    if (patterns.length > 0) {
        for (var _i = 0, patterns_1 = patterns; _i < patterns_1.length; _i++) {
            var regexp = patterns_1[_i];
            if (path.match(regexp)) {
                return path + suffix;
            }
        }
    }
    return path;
}
exports.appendSuffixIfMatch = appendSuffixIfMatch;
function appendSuffixesIfMatch(suffixDict, path) {
    for (var suffix in suffixDict) {
        path = appendSuffixIfMatch(suffixDict[suffix], path, suffix);
    }
    return path;
}
exports.appendSuffixesIfMatch = appendSuffixesIfMatch;
/**
 * Recursively collect all possible dependants of passed file
 */
function collectAllDependants(reverseDependencyGraph, fileName, collected) {
    if (collected === void 0) { collected = {}; }
    var result = {};
    result[fileName] = true;
    collected[fileName] = true;
    var dependants = reverseDependencyGraph[fileName];
    if (dependants !== undefined) {
        Object.keys(dependants).forEach(function (dependantFileName) {
            if (!collected[dependantFileName]) {
                collectAllDependants(reverseDependencyGraph, dependantFileName, collected)
                    .forEach(function (fName) { return result[fName] = true; });
            }
        });
    }
    return Object.keys(result);
}
exports.collectAllDependants = collectAllDependants;
/**
 * Recursively collect all possible dependencies of passed file
 */
function collectAllDependencies(dependencyGraph, filePath, collected) {
    if (collected === void 0) { collected = {}; }
    var result = {};
    result[filePath] = true;
    collected[filePath] = true;
    var directDependencies = dependencyGraph[filePath];
    if (directDependencies !== undefined) {
        directDependencies.forEach(function (dependencyModule) {
            if (!collected[dependencyModule.originalFileName]) {
                collectAllDependencies(dependencyGraph, dependencyModule.resolvedFileName, collected)
                    .forEach(function (filePath) { return result[filePath] = true; });
            }
        });
    }
    return Object.keys(result);
}
exports.collectAllDependencies = collectAllDependencies;
function arrify(val) {
    if (val === null || val === undefined) {
        return [];
    }
    return Array.isArray(val) ? val : [val];
}
exports.arrify = arrify;
;
