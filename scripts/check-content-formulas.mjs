import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import katex from 'katex';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(root, 'output/content-rebuild');
const pages = JSON.parse(fs.readFileSync(path.join(output, 'inventory.json'), 'utf8')).pages;
const problems = [];
let formulas = 0;
let articles = 0;

for (const page of pages) {
  const file = path.join(output, 'pages', page.id, 'reviewed.json');
  if (!fs.existsSync(file)) continue;
  articles += 1;
  let body = JSON.parse(fs.readFileSync(file, 'utf8')).body_markdown
    .replace(/^```[^\n]*\n[\s\S]*?^```\s*$/gm, '');
  const values = [];
  body = body.replace(/\$\$([\s\S]+?)\$\$/g, (_, source) => {
    values.push({ source, displayMode: true });
    return '';
  });
  for (const match of body.matchAll(/(?<![\\$])\$(?!\$)([^\n$]+?)(?<!\\)\$(?!\$)/g)) {
    values.push({ source: match[1], displayMode: false });
  }
  for (const formula of values) {
    formulas += 1;
    try {
      katex.renderToString(formula.source, {
        displayMode: formula.displayMode, throwOnError: true, strict: 'ignore', trust: false,
      });
    } catch (error) {
      problems.push({ path: page.path, source: formula.source, message: error.message });
    }
  }
}

const result = { articles, formulas, problems };
fs.writeFileSync(path.join(output, 'formula-check.json'), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ articles, formulas, errors: problems.length, problems }, null, 2));
process.exitCode = problems.length ? 1 : 0;
