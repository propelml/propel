/* A custom AST walker for documentation. This was written because
 - TypeDoc is unable to generate documentation for a single exported module, as
   we have with api.ts,
 - TypeDoc has an unreasonable amount of dependencies and code,
 - we want very nice looking documentation without superfluous junk. This gives
   full control.
*/
// tslint:disable:object-literal-sort-keys
import * as fs from "fs";
import * as path from "path";
import * as ts from "typescript";
import { assert } from "./util";

export interface DocEntry {
  kind: "class" | "method" | "property";
  name: string;
  typestr?: string;
  docstr?: string;
  args?: ArgEntry[];
  retType?: string;
}

export interface ArgEntry {
  name: string;
  typestr?: string;
  docstr?: string;
}

function toHTMLIndex(docs: DocEntry[]): string {
  let out = `<ol class="docindex">\n`;
  for (const entry of docs) {
    out += `<li>${entry.name}</li>\n`;
  }
  out += `</ol>\n`;
  return out;
}

function htmlBody(inner: string): string {
  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Propel Docs</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://fonts.googleapis.com/css?family=Nunito" rel="stylesheet">
  <link rel="stylesheet" href="normalize.css">
  <link rel="stylesheet" href="skeleton.css">
  <link rel="stylesheet" href="codemirror.css"/>
  <link rel="stylesheet" href="style.css">
  <link rel="icon" type="image/png" href="favicon.png">
  <script src="dist/notebook.js"></script>
</head>
  <body>${inner}</body>
</html>
  `;
}

export function htmlEntry(entry: DocEntry): string {
  let out = `<h2 class="name">${entry.name}</h2>\n`;
  if (entry.typestr) {
    out += `<div class="typestr">${entry.typestr}</div>\n`;
  }
  out += `<p>${entry.kind}</p>\n`;

  if (entry.docstr) {
    out += `<p class="docstr">${entry.docstr}</p>\n`;
  }

  if (entry.args && entry.args.length > 0) {
    out += `<p>Arguments:</p> <ol class="args">\n`;
    for (const arg of entry.args) {
      out += `<li>\n`;
      out += `<span class="name">${arg.name}</span>\n`;
      out += `<span class="typestr">${arg.typestr}</span>\n`;
      if (arg.docstr) {
        out += `<span class="docstr">${arg.docstr}</span>\n`;
      }
      out += `</li>\n`;
    }
    out += `</ol>\n`;
  }
  if (entry.retType) {
    out += `<p>Returns:</p>`;
    out += `<span class="retType">${entry.retType}</span>\n`;
  }
  return out;
}

export function toHTML(docs: DocEntry[]): string {
  let out = "";

  out += `<div class="panel">\n`;
  out += `<h1>Propel</h1>\n`;
  out += toHTMLIndex(docs);
  out += `</div>\n`;

  out += `<div class="container doc-entries">\n`;
  for (const entry of docs) {
    out += "<div class=\"doc-entry\">\n";
    out += htmlEntry(entry);
    out += "</div>\n";
  }
  out += `</div>\n`;
  return htmlBody(out);
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
      console.error("requestVisit", s.getName());
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
      if (sourceFile.fileName.endsWith(rootFile)) {
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
      const symbol = checker.getSymbolAtLocation(node.name);
      // checker.typeToString
      // checker.symbolToString
      // console.error("- type alias", checker.typeToString(node.type));
      // console.error(""); // New Line.
    } else if (ts.isStringLiteral(node)) {
      console.error("- string literal");
      console.error(""); // New Line.
    } else if (ts.isVariableDeclaration(node)) {
      const symbol = checker.getSymbolAtLocation(node.name);
      const name = symbol.getName();
      if (ts.isFunctionLike(node.initializer)) {
        visitMethod(node.initializer, name);
      } else {
        console.error("- var", name);
        console.error(""); // New Line.
      }
    } else if (ts.isFunctionDeclaration(node)) {
      const symbol = checker.getSymbolAtLocation(node.name);
      visitMethod(node, symbol.getName());

    } else if (ts.isFunctionTypeNode(node)) {
      console.error("- FunctionTypeNode.. ?");

    } else if (ts.isFunctionExpression(node)) {
      const symbol = checker.getSymbolAtLocation(node.name);
      const name = symbol ? symbol.getName() : "<unknown>";
      console.error("- FunctionExpression", name);

    } else if (ts.isInterfaceDeclaration(node)) {
      const symbol = checker.getSymbolAtLocation(node.name);
      const name = symbol.getName();
      console.error("- Interface", name);

    } else if (ts.isObjectLiteralExpression(node)) {
      // TODO Ignoring for now.
      console.error("- ObjectLiteralExpression");

    } else {
      console.log("Unknown node", node.kind);
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
    const name = className ? `${className}.${methodName}` : methodName;

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
    });
  }

  function getFlatDocstr(sym: ts.Symbol): string | undefined {
    if (sym && sym.getDocumentationComment().length > 0) {
      return ts.displayPartsToString(sym.getDocumentationComment());
    }
    return undefined;
  }

  function visitClass(node: ts.ClassDeclaration) {
    const symbol = checker.getSymbolAtLocation(node.name);
    const className = symbol.getName();

    let docstr = null;
    if (symbol.getDocumentationComment().length > 0) {
      docstr = ts.displayPartsToString(symbol.getDocumentationComment());
    }
    output.push({
      name: className,
      kind: "class",
      docstr,
    });

    for (const m of node.members) {
      const name = classElementName(m);

      // Skip private members.
      if (ts.getCombinedModifierFlags(m) & ts.ModifierFlags.Private) {
        console.error("private. skipping", name);
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
        console.log("member", className, name);
        console.log(""); // New Line.
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
    });
  }

  function classElementName(m: ts.ClassElement): string {
    return m.name && ts.isIdentifier(m.name) ? m.name.text : "<unknown>";
  }

  // TODO use tsconfig.json instead of supplying config.
  gen("api.ts", {
    target: ts.ScriptTarget.ES5, module: ts.ModuleKind.CommonJS
  });

  // console.log(JSON.stringify(output, null, 2));
  return output;
}

function writeHTML() {
  const docs = genJSON();
  console.log(JSON.stringify(docs, null, 2));
  const html = toHTML(docs);
  const fn = path.resolve(__dirname, "website/docs.html");
  fs.writeFileSync(fn, html);
  console.log("Wrote", fn);
}

if (require.main === module) {
  writeHTML();
}
