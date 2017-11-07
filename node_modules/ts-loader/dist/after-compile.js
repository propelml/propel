"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var path = require("path");
var utils_1 = require("./utils");
var constants = require("./constants");
function makeAfterCompile(instance, configFilePath) {
    var getCompilerOptionDiagnostics = true;
    var checkAllFilesForErrors = true;
    return function (compilation, callback) {
        // Don't add errors for child compilations
        if (compilation.compiler.isChild()) {
            callback();
            return;
        }
        removeTSLoaderErrors(compilation.errors);
        provideCompilerOptionDiagnosticErrorsToWebpack(getCompilerOptionDiagnostics, compilation, instance, configFilePath);
        getCompilerOptionDiagnostics = false;
        var modules = determineModules(compilation);
        var filesToCheckForErrors = determineFilesToCheckForErrors(checkAllFilesForErrors, instance);
        checkAllFilesForErrors = false;
        var filesWithErrors = {};
        provideErrorsToWebpack(filesToCheckForErrors, filesWithErrors, compilation, modules, instance);
        provideDeclarationFilesToWebpack(filesToCheckForErrors, instance.languageService, compilation);
        instance.filesWithErrors = filesWithErrors;
        instance.modifiedFiles = null;
        callback();
    };
}
exports.makeAfterCompile = makeAfterCompile;
/**
 * handle compiler option errors after the first compile
 */
function provideCompilerOptionDiagnosticErrorsToWebpack(getCompilerOptionDiagnostics, compilation, instance, configFilePath) {
    if (getCompilerOptionDiagnostics) {
        var languageService = instance.languageService, loaderOptions = instance.loaderOptions, compiler = instance.compiler;
        utils_1.registerWebpackErrors(compilation.errors, utils_1.formatErrors(languageService.getCompilerOptionsDiagnostics(), loaderOptions, instance.colors, compiler, { file: configFilePath || 'tsconfig.json' }));
    }
}
/**
 * build map of all modules based on normalized filename
 * this is used for quick-lookup when trying to find modules
 * based on filepath
 */
function determineModules(compilation) {
    var modules = {};
    compilation.modules.forEach(function (module) {
        if (module.resource) {
            var modulePath = path.normalize(module.resource);
            if (utils_1.hasOwnProperty(modules, modulePath)) {
                var existingModules = modules[modulePath];
                if (existingModules.indexOf(module) === -1) {
                    existingModules.push(module);
                }
            }
            else {
                modules[modulePath] = [module];
            }
        }
    });
    return modules;
}
function determineFilesToCheckForErrors(checkAllFilesForErrors, instance) {
    var files = instance.files, modifiedFiles = instance.modifiedFiles, filesWithErrors = instance.filesWithErrors;
    // calculate array of files to check
    var filesToCheckForErrors = {};
    if (checkAllFilesForErrors) {
        // check all files on initial run
        filesToCheckForErrors = files;
    }
    else if (modifiedFiles !== null && modifiedFiles !== undefined) {
        // check all modified files, and all dependants
        Object.keys(modifiedFiles).forEach(function (modifiedFileName) {
            utils_1.collectAllDependants(instance.reverseDependencyGraph, modifiedFileName)
                .forEach(function (fileName) {
                filesToCheckForErrors[fileName] = files[fileName];
            });
        });
    }
    // re-check files with errors from previous build
    if (filesWithErrors !== undefined) {
        Object.keys(filesWithErrors).forEach(function (fileWithErrorName) {
            return filesToCheckForErrors[fileWithErrorName] = filesWithErrors[fileWithErrorName];
        });
    }
    return filesToCheckForErrors;
}
function provideErrorsToWebpack(filesToCheckForErrors, filesWithErrors, compilation, modules, instance) {
    var compiler = instance.compiler, languageService = instance.languageService, files = instance.files, loaderOptions = instance.loaderOptions, compilerOptions = instance.compilerOptions;
    var filePathRegex = !!compilerOptions.checkJs ? constants.dtsTsTsxJsJsxRegex : constants.dtsTsTsxRegex;
    Object.keys(filesToCheckForErrors)
        .filter(function (filePath) { return filePath.match(filePathRegex); })
        .forEach(function (filePath) {
        var errors = languageService.getSyntacticDiagnostics(filePath).concat(languageService.getSemanticDiagnostics(filePath));
        if (errors.length > 0) {
            filesWithErrors[filePath] = files[filePath];
        }
        // if we have access to a webpack module, use that
        if (utils_1.hasOwnProperty(modules, filePath)) {
            var associatedModules = modules[filePath];
            associatedModules.forEach(function (module) {
                // remove any existing errors
                removeTSLoaderErrors(module.errors);
                // append errors
                var formattedErrors = utils_1.formatErrors(errors, loaderOptions, instance.colors, compiler, { module: module });
                utils_1.registerWebpackErrors(module.errors, formattedErrors);
                utils_1.registerWebpackErrors(compilation.errors, formattedErrors);
            });
        }
        else {
            // otherwise it's a more generic error
            utils_1.registerWebpackErrors(compilation.errors, utils_1.formatErrors(errors, loaderOptions, instance.colors, compiler, { file: filePath }));
        }
    });
}
/**
 * gather all declaration files from TypeScript and output them to webpack
 */
function provideDeclarationFilesToWebpack(filesToCheckForErrors, languageService, compilation) {
    Object.keys(filesToCheckForErrors)
        .filter(function (filePath) { return filePath.match(constants.tsTsxRegex); })
        .forEach(function (filePath) {
        var output = languageService.getEmitOutput(filePath);
        var declarationFile = output.outputFiles.filter(function (outputFile) { return outputFile.name.match(constants.dtsDtsxRegex); }).pop();
        if (declarationFile !== undefined) {
            var assetPath = path.relative(compilation.compiler.context, declarationFile.name);
            compilation.assets[assetPath] = {
                source: function () { return declarationFile.text; },
                size: function () { return declarationFile.text.length; },
            };
        }
    });
}
/**
 * handle all other errors. The basic approach here to get accurate error
 * reporting is to start with a "blank slate" each compilation and gather
 * all errors from all files. Since webpack tracks errors in a module from
 * compilation-to-compilation, and since not every module always runs through
 * the loader, we need to detect and remove any pre-existing errors.
 */
function removeTSLoaderErrors(errors) {
    var index = -1;
    var length = errors.length;
    while (++index < length) {
        if (errors[index].loaderSource === 'ts-loader') {
            errors.splice(index--, 1);
            length--;
        }
    }
}
