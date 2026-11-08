// Builds the React app into ../static (served by app/server.py; committed, so Node is only needed to change the UI).
//   npm install && npm run build      one-off build
//   npm run dev                        rebuild on every save (keep `python -m app` running)
import * as esbuild from "esbuild";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const out = resolve(here, "../static");
mkdirSync(resolve(out, "assets"), { recursive: true });
writeFileSync(resolve(out, "index.html"), readFileSync(resolve(here, "index.html"), "utf8"));

const watch = process.argv.includes("--watch");
const options = {
  entryPoints: { app: resolve(here, "src/main.jsx") },
  bundle: true,
  minify: !watch,
  sourcemap: watch ? "inline" : false,
  outdir: resolve(out, "assets"),
  format: "esm",
  target: ["es2020"],
  jsx: "automatic",
  loader: { ".js": "jsx" },
  define: { "process.env.NODE_ENV": watch ? '"development"' : '"production"' },
  nodePaths: process.env.NODE_PATH ? process.env.NODE_PATH.split(":") : [],
  logLevel: "info",
};
if (watch) {
  const ctx = await esbuild.context(options);
  await ctx.watch();
  console.log("watching src/ …");
} else {
  await esbuild.build(options);
}
