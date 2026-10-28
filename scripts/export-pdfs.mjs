#!/usr/bin/env node

import { createReadStream, existsSync } from "node:fs";
import { mkdir, readdir, stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, join, relative, resolve, sep } from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";

import { zhSidebar } from "../src/.vuepress/sidebar/zh.ts";

const projectRoot = resolve(fileURLToPath(new URL("..", import.meta.url)));
const sourceRoot = join(projectRoot, "src");
const distRoot = join(sourceRoot, ".vuepress", "dist");

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".gif": "image/gif",
  ".html": "text/html; charset=utf-8",
  ".ico": "image/x-icon",
  ".jpeg": "image/jpeg",
  ".jpg": "image/jpeg",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".webp": "image/webp",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
};

const options = parseArgs(process.argv.slice(2));

if (options.help) {
  printHelp();
  process.exit(0);
}

const groups = await collectGroups();

if (options.list) {
  for (const group of groups) {
    console.log(`${group.category} / ${group.title}\t${group.link}\t${group.pages.length} 篇`);
  }
  process.exit(0);
}

const selectedGroups = selectGroups(groups, options.only);
if (selectedGroups.length === 0) {
  throw new Error(`没有找到匹配的二级目录：${options.only}`);
}

if (!options.skipBuild) {
  await run("pnpm", ["docs:build"]);
} else if (!existsSync(distRoot)) {
  throw new Error("构建目录不存在，请移除 --skip-build 后重试");
}

const outputRoot = resolve(projectRoot, options.output);
await mkdir(outputRoot, { recursive: true });

const server = await startStaticServer(distRoot);
let browser;

try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    colorScheme: "light",
    javaScriptEnabled: false,
    viewport: { width: 1440, height: 1000 },
  });

  for (const [index, group] of selectedGroups.entries()) {
    const label = `[${index + 1}/${selectedGroups.length}]`;
    console.log(`${label} 正在导出 ${group.title}（${group.pages.length} 篇）`);
    const categoryDirectory = join(outputRoot, sanitizeFilename(group.category));
    await mkdir(categoryDirectory, { recursive: true });
    const outputFile = join(categoryDirectory, `${sanitizeFilename(group.title)}.pdf`);
    await exportGroup(page, server.origin, group, outputFile, options.watermark);
    console.log(`${label} 已生成 ${relative(projectRoot, outputFile)}`);
  }
} catch (error) {
  if (String(error?.message).includes("Executable doesn't exist")) {
    throw new Error(
      "未找到 Chromium。请先运行 `pnpm exec playwright install chromium`，再重新导出。",
      { cause: error },
    );
  }
  throw error;
} finally {
  await browser?.close();
  await server.close();
}

async function collectGroups() {
  const result = [];

  for (const entries of Object.values(zhSidebar)) {
    for (const entry of entries) {
      for (const child of entry.children ?? []) {
        if (!isGroup(child)) continue;

        const directory = join(sourceRoot, child.link);
        if (!existsSync(directory) || !(await stat(directory)).isDirectory()) continue;

        const configuredPages = child.children
          .map((page) => (typeof page === "string" ? page : page?.link))
          .filter((page) => typeof page === "string" && page.endsWith(".md"))
          .map((page) => toPageLink(child.link, page));

        const configuredFiles = new Set(
          child.children
            .map((page) => (typeof page === "string" ? page : page?.link))
            .filter((page) => typeof page === "string")
            .map((page) => page.split("/").at(-1)),
        );

        const extraPages = (await readdir(directory, { withFileTypes: true }))
          .filter(
            (file) =>
              file.isFile() &&
              file.name.endsWith(".md") &&
              file.name !== "README.md" &&
              !configuredFiles.has(file.name),
          )
          .map((file) => toPageLink(child.link, file.name))
          .sort((a, b) => a.localeCompare(b, "zh-CN"));

        result.push({
          category: entry.text,
          title: child.text,
          link: normalizeRoute(child.link),
          pages: [normalizeRoute(child.link), ...configuredPages, ...extraPages],
        });
      }
    }
  }

  return result;
}

function isGroup(value) {
  return (
    value &&
    typeof value === "object" &&
    typeof value.text === "string" &&
    typeof value.link === "string" &&
    value.link.endsWith("/") &&
    Array.isArray(value.children)
  );
}

function toPageLink(groupLink, pageLink) {
  if (pageLink.startsWith("/")) return normalizeRoute(pageLink);
  const htmlLink = pageLink.replace(/README\.md$/i, "").replace(/\.md$/i, ".html");
  return normalizeRoute(`${groupLink}${htmlLink}`);
}

function normalizeRoute(route) {
  const normalized = `/${route}`.replaceAll("//", "/");
  return normalized.endsWith(".md")
    ? normalized.replace(/\.md$/i, ".html")
    : normalized;
}

function selectGroups(allGroups, only) {
  if (!only) return allGroups;
  const needle = only.replace(/^\//, "").replace(/\/$/, "").toLowerCase();
  return allGroups.filter((group) => {
    const route = group.link.replace(/^\//, "").replace(/\/$/, "").toLowerCase();
    return route === needle || group.title.toLowerCase() === needle;
  });
}

async function exportGroup(page, origin, group, outputFile, watermark) {
  const articles = [];

  for (const [pageIndex, route] of group.pages.entries()) {
    console.log(`  - [${pageIndex + 1}/${group.pages.length}] 读取 ${route}`);
    const response = await page.goto(`${origin}${encodeURI(route)}`, {
      waitUntil: "load",
    });
    if (!response?.ok()) {
      throw new Error(`页面加载失败 (${response?.status() ?? "unknown"})：${route}`);
    }

    await page.waitForSelector(".vp-page", { state: "attached" });
    await waitForImages(page);
    console.log(`    页面资源就绪`);

    const article = await page.evaluate(() => {
      const source = document.querySelector(".vp-page");
      if (!source) throw new Error("找不到 .vp-page 正文容器");

      const clone = source.cloneNode(true);
      clone
        .querySelectorAll(
          ".vp-page-nav, .vp-page-meta, .read-more-lock, .page-author-info, .page-info",
        )
        .forEach((element) => element.remove());
      clone.querySelectorAll("[style]").forEach((element) => {
        if (element.style.display === "none") element.style.removeProperty("display");
      });
      clone.querySelectorAll("a.header-anchor").forEach((element) => {
        element.replaceWith(...element.childNodes);
      });

      const pageTitle = clone.querySelector(".vp-page-title h1");
      const markdownTitle = clone.querySelector("#markdown-content > h1:first-child");
      if (
        pageTitle?.textContent?.trim() &&
        pageTitle.textContent.trim() === markdownTitle?.textContent?.trim()
      ) {
        markdownTitle.remove();
      }

      return {
        html: clone.innerHTML,
        title: document.querySelector("h1")?.textContent?.trim() ?? document.title,
      };
    });

    articles.push(article);
  }

  console.log("  - 正在整理合并页面");
  await page.evaluate(
    ({ articles, groupTitle, watermark }) => {
      document.documentElement.dataset.theme = "light";
      document.title = groupTitle;

      document
        .querySelectorAll(
          ".vp-navbar, .vp-sidebar, .vp-footer-wrapper, .vp-breadcrumb, .toggle-sidebar-wrapper, .vp-skip-link",
        )
        .forEach((element) => element.remove());

      const pageRoot = document.querySelector(".vp-page");
      if (!pageRoot) throw new Error("找不到 PDF 正文容器");
      pageRoot.innerHTML = "";
      pageRoot.className = "vp-page pdf-book";

      const watermarkNode = document.createElement("div");
      watermarkNode.className = "pdf-watermark";
      watermarkNode.textContent = watermark;
      document.body.append(watermarkNode);

      for (const [index, article] of articles.entries()) {
        const section = document.createElement("article");
        section.className = `pdf-article ${index === 0 ? "pdf-introduction" : ""}`;
        section.dataset.title = article.title;
        section.innerHTML = article.html;
        pageRoot.append(section);
      }

      const style = document.createElement("style");
      style.textContent = `
        @page {
          size: A4;
          margin: 13mm 15mm 15mm;
        }

        html, body {
          background: #fff !important;
          color: #1f2937 !important;
          print-color-adjust: exact;
          -webkit-print-color-adjust: exact;
        }

        body::before, body::after { display: none !important; }
        #app { visibility: visible !important; }
        .theme-container {
          min-height: 0 !important;
          background: #fff !important;
          background-image: none !important;
        }

        .pdf-book {
          box-sizing: border-box;
          display: block !important;
          max-width: none !important;
          margin: 0 !important;
          padding: 0 !important;
          background: #fff !important;
          background-image: none !important;
        }

        .pdf-article {
          box-sizing: border-box;
          break-before: page;
          max-width: none;
          margin: 0;
          padding: 0;
          background: #fff !important;
          background-image: none !important;
        }

        .pdf-article:first-child { break-before: auto; }

        .pdf-book::before,
        .pdf-book::after,
        .pdf-article::before,
        .pdf-article::after {
          content: none !important;
          display: none !important;
          background: none !important;
        }

        .pdf-article + .pdf-article {
          margin-top: 20px;
          padding-top: 24px;
          border-top: 1px solid #dce3ec;
        }

        .pdf-article > main,
        .pdf-article .vp-page-main,
        .pdf-article .vp-doc,
        .pdf-article #markdown-content {
          box-sizing: border-box;
          max-width: none !important;
          margin: 0 !important;
          padding: 0 !important;
          background: #fff !important;
          background-image: none !important;
        }

        .pdf-article h1 {
          color: #111827 !important;
          font-size: 25px !important;
          line-height: 1.35 !important;
          margin: 0 0 14px !important;
          padding-top: 0 !important;
          border: 0 !important;
          letter-spacing: 0 !important;
          break-after: avoid-page;
        }

        .pdf-article h2 {
          color: #0b5db5 !important;
          font-size: 19px !important;
          line-height: 1.45 !important;
          margin: 21px 0 8px !important;
          letter-spacing: 0 !important;
          break-after: avoid-page;
        }

        .pdf-article h3 {
          color: #1f2937 !important;
          font-size: 16px !important;
          line-height: 1.5 !important;
          margin: 16px 0 6px !important;
          letter-spacing: 0 !important;
          break-after: avoid-page;
        }

        .pdf-article p,
        .pdf-article li {
          font-size: 12px !important;
          line-height: 1.65 !important;
          letter-spacing: 0 !important;
        }

        .pdf-article p { margin: 6px 0 !important; }
        .pdf-article figure { margin: 10px auto !important; }
        .pdf-article img {
          display: block;
          max-width: 100% !important;
          max-height: 165mm;
          margin: 9px auto !important;
          object-fit: contain;
          break-inside: avoid;
        }

        .pdf-article pre {
          break-inside: auto;
          white-space: pre-wrap !important;
          overflow-wrap: anywhere;
          font-size: 9.5px !important;
        }

        .pdf-article table {
          break-inside: auto;
          display: table !important;
          width: 100% !important;
          font-size: 10px !important;
        }

        .pdf-article a {
          color: #096dd9 !important;
          overflow-wrap: anywhere;
          text-decoration: none !important;
        }

        .pdf-watermark {
          position: fixed;
          top: 48%;
          left: 50%;
          z-index: 2147483647;
          color: rgba(9, 109, 217, 0.095);
          font-family: system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
          font-size: 44px;
          font-weight: 700;
          letter-spacing: 0;
          line-height: 1;
          pointer-events: none;
          transform: translate(-50%, -50%) rotate(-30deg);
          white-space: nowrap;
        }

        @media print {
          html, body, #app, .theme-container, .pdf-book, .pdf-article {
            background: #fff !important;
            background-image: none !important;
          }

          .pdf-article { break-before: page; }
          .pdf-article:first-child { break-before: auto; }
        }
      `;
      document.head.append(style);
    },
    { articles, groupTitle: group.title, watermark },
  );

  console.log("    合并页面已生成");
  await page.waitForTimeout(1000);
  console.log("    合并页面资源就绪");
  console.log("  - 正在写入 PDF");
  await page.emulateMedia({ media: "print" });
  await page.pdf({
    path: outputFile,
    format: "A4",
    printBackground: true,
    displayHeaderFooter: true,
    headerTemplate: "<span></span>",
    footerTemplate: `
      <div style="box-sizing:border-box;width:100%;padding:0 16mm;color:#8a94a3;font:9px system-ui,-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;text-align:right;">
        <span class="pageNumber"></span> / <span class="totalPages"></span>
      </div>
    `,
    margin: { top: "13mm", right: "15mm", bottom: "15mm", left: "15mm" },
    preferCSSPageSize: true,
  });
}

async function waitForImages(page) {
  await page.evaluate(async () => {
    const images = [...document.images];
    for (const image of images) image.loading = "eager";

    const imageSettled = Promise.all(
      images.map(
        (image) =>
          image.complete ||
          new Promise((resolve) => {
            image.addEventListener("load", resolve, { once: true });
            image.addEventListener("error", resolve, { once: true });
          }),
      ),
    );
    const timeout = (milliseconds) =>
      new Promise((resolve) => setTimeout(resolve, milliseconds));

    await Promise.race([imageSettled, timeout(5000)]);
    await Promise.race([document.fonts?.ready ?? Promise.resolve(), timeout(2000)]);
  });
}

async function startStaticServer(root) {
  const server = createServer(async (request, response) => {
    try {
      const requestUrl = new URL(request.url ?? "/", "http://127.0.0.1");
      let pathname = decodeURIComponent(requestUrl.pathname);
      if (pathname.endsWith("/")) pathname += "index.html";

      const filePath = resolve(root, `.${pathname}`);
      const relativePath = relative(root, filePath);
      if (relativePath.startsWith("..") || relativePath.split(sep).includes("..")) {
        response.writeHead(403).end("Forbidden");
        return;
      }

      const fileStat = await stat(filePath);
      if (!fileStat.isFile()) throw new Error("Not a file");

      response.writeHead(200, {
        "Content-Type": mimeTypes[extname(filePath).toLowerCase()] ?? "application/octet-stream",
        "Cache-Control": "no-cache",
      });
      createReadStream(filePath).pipe(response);
    } catch {
      response.writeHead(404).end("Not found");
    }
  });

  await new Promise((resolvePromise, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolvePromise);
  });

  const address = server.address();
  if (!address || typeof address === "string") throw new Error("无法启动本地静态服务器");

  return {
    origin: `http://127.0.0.1:${address.port}`,
    close: () => new Promise((resolvePromise) => server.close(resolvePromise)),
  };
}

function run(command, args) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, {
      cwd: projectRoot,
      stdio: "inherit",
      shell: process.platform === "win32",
    });
    child.once("error", reject);
    child.once("exit", (code) => {
      if (code === 0) resolvePromise();
      else reject(new Error(`${command} ${args.join(" ")} 执行失败，退出码 ${code}`));
    });
  });
}

function sanitizeFilename(value) {
  return value.replace(/[<>:"/\\|?*\u0000-\u001F]/g, "-").trim();
}

function parseArgs(args) {
  const parsed = {
    help: false,
    list: false,
    only: "",
    output: "output/pdf",
    skipBuild: false,
    watermark: "@安逸Ai",
  };

  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === "--") continue;
    if (argument === "--help" || argument === "-h") parsed.help = true;
    else if (argument === "--list") parsed.list = true;
    else if (argument === "--skip-build") parsed.skipBuild = true;
    else if (argument === "--only") parsed.only = requireValue(args, ++index, argument);
    else if (argument === "--output") parsed.output = requireValue(args, ++index, argument);
    else if (argument === "--watermark") parsed.watermark = requireValue(args, ++index, argument);
    else throw new Error(`未知参数：${argument}。运行 --help 查看用法。`);
  }

  return parsed;
}

function requireValue(args, index, argument) {
  const value = args[index];
  if (!value || value.startsWith("--")) throw new Error(`${argument} 缺少参数值`);
  return value;
}

function printHelp() {
  console.log(`
将 VuePress 二级目录中的文章合并为带水印的 PDF。

用法：
  pnpm docs:pdf
  pnpm docs:pdf -- --only frontier/agentic-rag

选项：
  --list                 列出所有可导出的二级目录
  --only <目录或中文名>  只导出一个目录，例如 frontier/agentic-rag
  --output <目录>        输出目录，默认 output/pdf
  --watermark <文本>     水印文字，默认 @安逸Ai
  --skip-build           使用现有的 VuePress 构建结果
  -h, --help             显示帮助
`);
}
