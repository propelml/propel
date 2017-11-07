"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var semver = require("semver");
var constants = require("./constants");
function getCompiler(loaderOptions, log) {
    var compiler;
    var errorMessage;
    var compilerDetailsLogMessage;
    var compilerCompatible = false;
    try {
        compiler = require(loaderOptions.compiler);
    }
    catch (e) {
        errorMessage = loaderOptions.compiler === 'typescript'
            ? 'Could not load TypeScript. Try installing with `yarn add typescript` or `npm install typescript`. If TypeScript is installed globally, try using `yarn link typescript` or `npm link typescript`.'
            : "Could not load TypeScript compiler with NPM package name `" + loaderOptions.compiler + "`. Are you sure it is correctly installed?";
    }
    if (errorMessage === undefined) {
        compilerDetailsLogMessage = "ts-loader: Using " + loaderOptions.compiler + "@" + compiler.version;
        compilerCompatible = false;
        if (loaderOptions.compiler === 'typescript') {
            if (compiler.version && semver.gte(compiler.version, '2.0.0')) {
                // don't log yet in this case, if a tsconfig.json exists we want to combine the message
                compilerCompatible = true;
            }
            else {
                log.logError(compilerDetailsLogMessage + ". This version is incompatible with ts-loader. Please upgrade to the latest version of TypeScript.");
            }
        }
        else {
            log.logWarning(compilerDetailsLogMessage + ". This version may or may not be compatible with ts-loader.");
        }
    }
    return { compiler: compiler, compilerCompatible: compilerCompatible, compilerDetailsLogMessage: compilerDetailsLogMessage, errorMessage: errorMessage };
}
exports.getCompiler = getCompiler;
function getCompilerOptions(configParseResult) {
    var compilerOptions = Object.assign({}, configParseResult.options, {
        skipLibCheck: true,
        suppressOutputPathCheck: true,
    });
    // if `module` is not specified and not using ES6+ target, default to CJS module output
    if ((compilerOptions.module === undefined) &&
        (compilerOptions.target !== undefined && compilerOptions.target < constants.ScriptTargetES2015)) {
        compilerOptions.module = constants.ModuleKindCommonJs;
    }
    return compilerOptions;
}
exports.getCompilerOptions = getCompilerOptions;
