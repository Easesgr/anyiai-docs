import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(root, 'output/content-rebuild');
const baseUrl = new URL(process.argv[2] || 'http://127.0.0.1:8399/');
if (!['127.0.0.1', 'localhost', '[::1]'].includes(baseUrl.hostname)) {
  throw new Error('Use the local documentation development server.');
}
const pages = JSON.parse(fs.readFileSync(path.join(output, 'inventory.json'), 'utf8')).pages;
const hash = (source) => crypto.createHash('sha256').update(source).digest('hex');
const results = [];
let index = 0;

await Promise.all(Array.from({ length: 4 }, async () => {
  while (index < pages.length) {
    const page = pages[index++];
    const source = path.join(root, 'src', page.path);
    try {
      const sourceHash = hash(fs.readFileSync(source));
      const url = new URL('/@fs' + source, baseUrl);
      url.searchParams.set('import', '');
      const response = await fetch(url, { signal: AbortSignal.timeout(60000) });
      const content = await response.text();
      let error;
      if (!response.ok) {
        const serialized = content.match(/const error = (.*)\n/);
        if (serialized) {
          const diagnostic = JSON.parse(serialized[1]);
          error = { message: diagnostic.message, location: diagnostic.loc, frame: diagnostic.frame };
        } else {
          error = { message: content.slice(0, 1000) };
        }
      }
      const current = hash(fs.readFileSync(source)) === sourceHash;
      results.push({
        path: page.path, source_sha256: sourceHash, status: response.status,
        compiled: response.ok && response.headers.get('content-type')?.includes('javascript') === true && current,
        ...(error ? { error } : {}),
        ...(!current ? { error: { message: 'Source changed during development rendering; check it again.' } } : {}),
      });
    } catch (error) {
      results.push({ path: page.path, compiled: false, error: { message: error.message } });
    }
  }
}));

results.sort((left, right) => left.path.localeCompare(right.path));
const failures = results.filter((result) => !result.compiled);
const report = { checked: results.length, base_url: baseUrl.href, failures, results };
fs.writeFileSync(path.join(output, 'dev-render-check.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ checked: results.length, failures }, null, 2));
process.exitCode = failures.length ? 1 : 0;
