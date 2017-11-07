"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var path = require("path");
var utils_1 = require("./utils");
var constants = require("./constants");
/**
 * Make function which will manually update changed files
 */
function makeWatchRun(instance) {
    var lastTimes = {};
    var startTime = null;
    return function (watching, cb) {
        if (null === instance.modifiedFiles) {
            instance.modifiedFiles = {};
        }
        startTime = startTime || watching.startTime;
        var times = watching.compiler.fileTimestamps;
        Object.keys(times)
            .filter(function (filePath) {
            return times[filePath] > (lastTimes[filePath] || startTime)
                && filePath.match(constants.tsTsxJsJsxRegex);
        })
            .forEach(function (filePath) {
            lastTimes[filePath] = times[filePath];
            filePath = path.normalize(filePath);
            var file = instance.files[filePath];
            if (file !== undefined) {
                file.text = utils_1.readFile(filePath) || '';
                file.version++;
                instance.version++;
                instance.modifiedFiles[filePath] = file;
            }
        });
        cb();
    };
}
exports.makeWatchRun = makeWatchRun;
