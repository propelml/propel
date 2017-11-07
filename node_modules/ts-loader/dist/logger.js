"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var console_1 = require("console");
var LogLevel;
(function (LogLevel) {
    LogLevel[LogLevel["INFO"] = 1] = "INFO";
    LogLevel[LogLevel["WARN"] = 2] = "WARN";
    LogLevel[LogLevel["ERROR"] = 3] = "ERROR";
})(LogLevel || (LogLevel = {}));
var stderrConsole = new console_1.Console(process.stderr);
var stdoutConsole = new console_1.Console(process.stdout);
var doNothingLogger = function (_message) { };
var makeLoggerFunc = function (loaderOptions) {
    return loaderOptions.silent
        ? function (_whereToLog, _message) { }
        : function (whereToLog, message) { return console.log.call(whereToLog, message); };
};
var makeExternalLogger = function (loaderOptions, logger) {
    return function (message) {
        return logger(loaderOptions.logInfoToStdOut ? stdoutConsole : stderrConsole, message);
    };
};
var makeLogInfo = function (loaderOptions, logger, green) {
    return LogLevel[loaderOptions.logLevel] <= LogLevel.INFO
        ? function (message) {
            return logger(loaderOptions.logInfoToStdOut ? stdoutConsole : stderrConsole, green(message));
        }
        : doNothingLogger;
};
var makeLogError = function (loaderOptions, logger, red) {
    return LogLevel[loaderOptions.logLevel] <= LogLevel.ERROR
        ? function (message) { return logger(stderrConsole, red(message)); }
        : doNothingLogger;
};
var makeLogWarning = function (loaderOptions, logger, yellow) {
    return LogLevel[loaderOptions.logLevel] <= LogLevel.WARN
        ? function (message) { return logger(stderrConsole, yellow(message)); }
        : doNothingLogger;
};
function makeLogger(loaderOptions, colors) {
    var logger = makeLoggerFunc(loaderOptions);
    return {
        log: makeExternalLogger(loaderOptions, logger),
        logInfo: makeLogInfo(loaderOptions, logger, colors.green),
        logWarning: makeLogWarning(loaderOptions, logger, colors.yellow),
        logError: makeLogError(loaderOptions, logger, colors.red)
    };
}
exports.makeLogger = makeLogger;
