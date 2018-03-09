/*!
   Copyright 2018 Propel http://propel.site/.  All rights reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */
/* A custom AST walker for documentation. This was written because
 - TypeDoc is unable to generate documentation for a single exported module, as
   we have with api.ts,
 - TypeDoc has an unreasonable amount of dependencies and code,
 - we want very nice looking documentation without superfluous junk. This gives
   full control.
*/
// tslint:disable:object-literal-sort-keys
import * as assert from "assert";
import { execSync, spawnSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import * as ts from "typescript";
import { log } from "../src/util";
import { ArgEntry, DocEntry } from "../website/docs";

const repoBasePath = path.resolve(__dirname, "..");
const repoBaseUrl = "https://github.com/propelml/propel";

const fileGithubUrls = new Map<string, string>();

function getGithubUrlForFile(fileName: string) {
  if (fileGithubUrls.has(fileName)) {
    return fileGithubUrls.get(fileName);
  }

  const relName = path.relative(repoBasePath, fileName).replace(/\\/g, "/");

  // Sanity check: verify that the file in it's current form has been
  // committed.
  let stdout = execSync(`git status --porcelain -- "${fileName}"`, {
    cwd: path.dirname(fileName),
    encoding: "utf8"
  });
  if (/\S/.test(stdout)) {
    throw new Error(`File has been modified since last commit: ${relName}.`);
  }

  // Get the commit hash for that most recent commit that updated a file.
  // This is done to reduce churn in the generated documentation; as long as a
  // file doesn't change, the "source" links in the documentation won't change
  // either.
  stdout = execSync(`git log -n1 --pretty="%H" -- "${fileName}"`, {
    cwd: path.dirname(fileName),
    encoding: "utf8"
  });
  const commitSha = stdout.match(/^\s*([0-9a-fA-F]{40})\s*$/)[1];
  const githubUrl = `${repoBaseUrl}/blob/${commitSha}/${relName}`;

  // Sanity check: verify that the inferred github url can actually be
  // loaded.
  const { status, stderr } = spawnSync(
    process.execPath,
    [`${__dirname}/check_url.js`, githubUrl],
    { encoding: "utf8" }
  );
  if (status !== 0) {
    const msg =
      `File committed but not available on github: ${relName}\n` +
      `You probably need to push your branch to github.\n` +
      stderr;
    console.warn(msg);
  }

  fileGithubUrls.set(fileName, githubUrl);
  return githubUrl;
}

export function genJSON(): DocEntry[] {
  // Global variables.
  const visitQueue: ts.Node[] = [];
  const visitHistory = new Map<ts.Symbol, boolean>();
  let checker: ts.TypeChecker = null;

  const output: DocEntry[] = [];

  function requestVisit(s: ts.Symbol) {
    if (!visitHistory.has(s)) {
      // Find original symbol (might not be in api.ts).
      s = skipAlias(s, checker);
      log("requestVisit", s.getName());
      const decls = s.getDeclarations();
      // What does it mean tot have multiple declarations?
      // assert(decls.length === 1);
      visitQueue.push(decls[0]);
      visitHistory.set(s, true);
    }
  }

  function requestVisitType(t: ts.Type) {
    if (t.symbol) {
      requestVisit(t.symbol);
    } else if (t.aliasSymbol) {
      requestVisit(t.aliasSymbol);
    }
  }

  function skipAlias(symbol: ts.Symbol, checker: ts.TypeChecker) {
    return symbol.flags & ts.SymbolFlags.Alias ?
      checker.getAliasedSymbol(symbol) : symbol;
  }

  /** Generate documentation for all classes in a set of .ts files */
  function gen(rootFile: string, options: ts.CompilerOptions): void {
    // Build a program using the set of root file names in fileNames
    const program = ts.createProgram([rootFile], options);

    // Get the checker, we will use it to find more about classes
    checker = program.getTypeChecker();

    // Find the SourceFile object corresponding to our rootFile.
    let rootSourceFile = null;
    for (const sourceFile of program.getSourceFiles()) {
      if (path.resolve(sourceFile.fileName) === path.resolve(rootFile)) {
        rootSourceFile = sourceFile;
        break;
      }
    }
    assert(rootSourceFile);

    // Add all exported symbols of root module to visitQueue.
    const moduleSymbol = checker.getSymbolAtLocation(rootSourceFile);
    for (const s of checker.getExportsOfModule(moduleSymbol)) {
      requestVisit(s);
    }

    // Process queue of Nodes that should be displayed in docs.
    while (visitQueue.length) {
      const n = visitQueue.shift();
      visit(n);
    }
  }

  // visit nodes finding exported classes
  function visit(node: ts.Node) {

    if (ts.isClassDeclaration(node) && node.name) {
      // This is a top level class, get its symbol
      visitClass(node);
    } else if (ts.isTypeAliasDeclaration(node)) {
      // const symbol = checker.getSymbolAtLocation(node.name);
      // checker.typeToString
      // checker.symbolToString
      // console.error("- type alias", checker.typeToString(node.type));
      // console.error(""); // New Line.
    } else if (ts.isStringLiteral(node)) {
      log("- string literal");
    } else if (ts.isVariableDeclaration(node)) {
      const symbol = checker.getSymbolAtLocation(node.name);
      const name = symbol.getName();
      if (ts.isFunctionLike(node.initializer)) {
        visitMethod(node.initializer, name);
      } else {
        log("- var", name);
      }
    } else if (ts.isFunctionDeclaration(node)) {
      const symbol = checker.getSymbolAtLocation(node.name);
      visitMethod(node, symbol.getName());

    } else if (ts.isFunctionTypeNode(node)) {
      log("- FunctionTypeNode.. ?");

    } else if (ts.isFunctionExpression(node)) {
      const symbol = checker.getSymbolAtLocation(node.name);
      const name = symbol ? symbol.getName() : "<unknown>";
      log("- FunctionExpression", name);

    } else if (ts.isInterfaceDeclaration(node)) {
      const symbol = checker.getSymbolAtLocation(node.name);
      const name = symbol.getName();
      log("- Interface", name);

    } else if (ts.isObjectLiteralExpression(node)) {
      // TODO Ignoring for now.
      log("- ObjectLiteralExpression");

    } else if (ts.isTypeLiteralNode(node)) {
      // TODO Ignoring for now.
      log("- TypeLiteral");

    } else {
      log("Unknown node", node.kind);
      assert(false);
    }
  }

  function visitMethod(methodNode: ts.FunctionLike,
                       methodName: string, className?: string) {
    // Get the documentation string.
    const sym = checker.getSymbolAtLocation(methodNode.name);
    const docstr = getFlatDocstr(sym);

    const sig = checker.getSignatureFromDeclaration(methodNode);
    const sigStr = checker.signatureToString(sig);
    let name;
    if (!className) {
      name = methodName;
    } else if (methodName.startsWith("[")) {
      // EG [Symbol.iterator]
      name = className + methodName;
    } else {
      name = `${className}.${methodName}`;
    }

    // Print each of the parameters.
    const argEntries: ArgEntry[] = [];
    for (const paramSymbol of sig.parameters) {
      const paramType = checker.getTypeOfSymbolAtLocation(paramSymbol,
        paramSymbol.valueDeclaration!);
      requestVisitType(paramType);

      argEntries.push({
        name: paramSymbol.getName(),
        typestr: checker.typeToString(paramType),
        docstr: getFlatDocstr(paramSymbol),
      });
    }

    const retType = sig.getReturnType();
    requestVisitType(retType);

    output.push({
      name,
      kind: "method",
      typestr: sigStr,
      args: argEntries,
      retType: checker.typeToString(retType),
      docstr,
      sourceUrl: getSourceUrl(methodNode)
    });
  }

  function getFlatDocstr(sym: ts.Symbol): string | undefined {
    if (sym && sym.getDocumentationComment(checker).length > 0) {
      return ts.displayPartsToString(sym.getDocumentationComment(checker));
    }
    return undefined;
  }

  function getSourceUrl(node: ts.Node): string {
    const sourceFile = node.getSourceFile();
    const docNodes = (node as any).jsDoc; // No public API for this?
    const startNode = (docNodes && docNodes[0]) || node;
    const [startLine, endLine] = [
      startNode.getStart(),
      node.getEnd()
    ].map(pos => sourceFile.getLineAndCharacterOfPosition(pos).line + 1);
    const sourceRange =
      endLine > startLine ? `L${startLine}-L${endLine}` : `L${startLine}`;
    const githubUrl = getGithubUrlForFile(sourceFile.fileName);
    return `${githubUrl}#${sourceRange}`;
  }

  function visitClass(node: ts.ClassDeclaration) {
    const symbol = checker.getSymbolAtLocation(node.name);
    const className = symbol.getName();

    let docstr = null;
    if (symbol.getDocumentationComment(checker).length > 0) {
      docstr = ts.displayPartsToString(symbol.getDocumentationComment(checker));
    }
    output.push({
      name: className,
      kind: "class",
      docstr,
      sourceUrl: getSourceUrl(node)
    });

    for (const m of node.members) {
      const name = classElementName(m);

      // Skip private members.
      if (ts.getCombinedModifierFlags(m) & ts.ModifierFlags.Private) {
        log("private. skipping", name);
        continue;
      }

      if (ts.isConstructorDeclaration(m)) {
        visitMethod(m, "constructor", className);

      } else if (ts.isMethodDeclaration(m)) {
        visitMethod(m, name, className);

      } else if (ts.isPropertyDeclaration(m)) {
        if (ts.isFunctionLike(m.initializer)) {
          visitMethod(m.initializer, name, className);
        } else {
          visitProp(m, name, className);
        }
      } else if (ts.isGetAccessorDeclaration(m)) {
        visitProp(m, name, className);

      } else {
        log("member", className, name);
      }
    }
  }

  function visitProp(node: ts.ClassElement, name: string, className?: string) {
    name = className ? `${className}.${name}` : name;

    const symbol = checker.getSymbolAtLocation(node.name);
    const t = checker.getTypeOfSymbolAtLocation(symbol, node);

    output.push({
      name,
      kind: "property",
      typestr: checker.typeToString(t),
      docstr: getFlatDocstr(symbol),
      sourceUrl: getSourceUrl(node)
    });
  }

  function classElementName(m: ts.ClassElement): string {
    if (m.name) {
      if (ts.isIdentifier(m.name)) {
        return ts.idText(m.name);
      }
      if (ts.isComputedPropertyName(m.name)) {
        const e = m.name.expression;
        if (ts.isPropertyAccessExpression(e)) {
          // This is for [Symbol.iterator]() { }
          assert(ts.isIdentifier(e.name));
          return `[Symbol.${e.name.text}]`;
        }
      }
    }
    return "<unknown>";
  }

  gen(repoBasePath + "/src/api.ts", require("../tsconfig.json"));

  return output;
}

export function writeJSON(target = repoBasePath + "/build/website") {
  const docs = genJSON();
  const j = JSON.stringify(docs, null, 2);
  fs.writeFileSync(target, j);
  console.log("wrote", target);
}

if (require.main === module) {
  const target = process.argv[2];
  if (!target) {
    console.log("Usage: ts-node tools/gendoc.ts ./website/docs.json");
    process.exit(1);
  }
  writeJSON(target);
}
