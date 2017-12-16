import * as ts from "typescript";
import * as dl from "./dl";

const WEB = typeof window !== "undefined";
const NODE = !WEB;
const nodeRequire = WEB ? null : ((mod) => require(mod));

const w3fetch = WEB ? window.fetch : nodeRequire("node-fetch");
const W3URL = WEB ? window.URL : require("url").URL;

let replBaseHref;
if (WEB) {
  replBaseHref = window.location;
} else {
  const url = new W3URL("file://");
  url.pathname = __filename;
  replBaseHref = url.href;
}

const BUILTINS = {
  seedrandom: require("seedrandom")
};

function resolveImport(href, baseHref) {
  const url = new W3URL(href, new W3URL(baseHref));
  return url.href;
}

function hrefGetPath(href) {
   const url = new W3URL(href);
   let p = url.pathname;
   if (url.protocol === "file:" &&
       process.platform === "win32" && /^\/[a-zA-Z]:/.test(p)) {
     // Remove leading slash before '/c:/windows/file.txt'
     p = p.slice(1);
   }
   return p;
}

export function fileext(s) {
  return /(?:\.([^\/\\.]*))?$/.exec(s)[1] || "";
}

async function getModuleSource(href) {
  if (NODE) {
    const url = new W3URL(href);
    if (url.protocol === "file:") {
      let fileName = url.pathname;
      if (process.platform === "win32" && /^\/[a-zA-Z]:/.test(fileName)) {
        // Remove leading slash before '/c:/windows/file.txt'
        fileName = fileName.slice(1);
      }
      return require("fs").readFileSync(fileName, "utf8");
    }
  }
  console.log("fetching " + href);
  const res = await w3fetch(href, { mode: "no-cors" });
  if (!res.ok) {
    throw new Error("Fetch failed: " + JSON.stringify(res));
  }
  return await res.text();
}

function getRelativeImports(code, ext) {
  const imports = [];
  const kind = (ext === "ts") ? ts.ScriptKind.TS
                              : ts.ScriptKind.JS;
  const sourceFile = ts.createSourceFile(
    "eval.ts",
    code,
    ts.ScriptTarget.ES2015,
    false, // setParentNodes
    kind);
  walk(sourceFile);
  return imports;

  function walk(node) {
    if (node.moduleSpecifier)  {
      imports.push(node.moduleSpecifier.text);
    }
    if (node.moduleReference) {
      console.log(node);
    }
    ts.forEachChild(node, walk);
  }
}

function getImports(code, baseHref, ext) {
  return getRelativeImports(code, ext)
    .filter(href => !/^\w*$/.test(href))
    .map(href => resolveImport(href, baseHref));
}

function transpile(code) {
  const compilerOptions = { diagnostics: true,
                            noImplicitUseStrict: true,
                            sourceMap: false };
  const tr = ts.transpileModule(code, { compilerOptions });
  return tr.outputText;
}

const wrapper =
   `let __$code = yield;
   for (;;) {
     try {
       __$code = yield { result: eval(__$code) };
     } catch (error) {
       __$code = yield { error }
     }
   }`;
// Works in javascript proper but ts-node somehow breaks it:
//  const Generator = Object.getPrototypeOf(function*(){}).constructor;
//  const evalScope = new Generator('exports', 'require', 'module', wrapper);
const src = `(function*(exports, require, module){${wrapper}})`;
const globalEval = eval;
const evalScope = globalEval(src);

export function Context() {
  const importSources = {};
  const importModules = {};

  async function fetchAndLoadRecursiveImports(href) {
    let ext = fileext(href);
    const tries = (ext === "ts" || ext === "js")
                   ? [ href ]
                   : [ `${href}.ts`, `${href}.js` ];
    let code, error;
    for (href of tries) {
      console.log(href);
      try {
        code = await getModuleSource(href);
        error = undefined;
        ext = fileext(href);
        break;
      } catch (e) {
        error = e;
      }
    }

    if (error) {
      throw error;
    }

    const imports = getImports(code, href, ext);
    await preloadImports(imports);

    if (ext === "ts") {
      code = transpile(code);
    }

    return code;
  }

  async function preloadImport(url) {
    if (typeof importSources[url] === "string") {
      return;
    } // Already loaded.

    if (importSources[url] instanceof Promise) {
      // TODO: the module is already loading, but can't await here because
      // imports may be circular, in which case we'd end up with two
      // async functions awaiting each other.
      // However there are scenarios where we would need to await here.
      return;
    }

    const promise = fetchAndLoadRecursiveImports(url);
    importSources[url] = promise;
    importSources[url] = await promise;
  }

  async function preloadImports(imports) {
    const promises = imports.map((url) => preloadImport(url));
    await Promise.all(promises);
  }

  function requireHelper(href, base) {
    if (href in BUILTINS) {
      return BUILTINS[href];
    }

    href = resolveImport(href, base);

    let module = importModules[href];
    if (module !== undefined) {
      return module.exports;
    }

    const code = importSources[href];
    if (typeof code !== "string") {
      if (WEB) {
        throw new Error(`Module source not available: ${href}\n` +
                        `    from ${base}`);
      } else {
        return nodeRequire(hrefGetPath(href));
      }
    }

    const exports = {};
    const require = makeRequireFunction(href);
    module = { exports, require };
    importModules[href] = module;

    const dirname = hrefGetPath(resolveImport(".", href));
    const filename = hrefGetPath(href);

    const fn = new Function("exports",
                            "require",
                            "module",
                            "__dirname",
                            "__filename",
                            code);
    fn(exports, require, module, dirname, filename);

    return module.exports;
  }

  function makeRequireFunction(base) {
    return (href) => requireHelper(href, base);
  }

  const replModule = {
    exports: {},
    require: makeRequireFunction(replBaseHref)
  };
  const scope = evalScope(replModule.exports, replModule.require, replModule);
  scope.eval = (code) => scope.next(code).value;
  scope.eval(); // Start the generator.

  this.eval = async(code) => {
    const js = transpile(code);
    const imports = getImports(code, replBaseHref, "ts");
    try {
      await preloadImports(imports);
    } catch (error) {
      return { error };
    }

    const { result, error } = scope.eval(js);
    return { result, error };
  };
}
