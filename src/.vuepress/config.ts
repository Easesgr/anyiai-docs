import { defineUserConfig } from "vuepress";
import { viteBundler } from "@vuepress/bundler-vite";

import theme from "./theme.js";

const googleSiteVerification = process.env.GOOGLE_SITE_VERIFICATION;

export default defineUserConfig({
  base: "/",

  lang: "zh-CN",
  title: "AI 学习资源站",
  description:
    "Anyi AI 是一个免费开源的 AI 学习资源站，提供系统化的 AI 学习路径。涵盖机器学习基础、深度学习、Transformer 架构、大模型（LLM）、RAG 检索增强生成、Agent 智能体、Prompt Engineering、模型微调（LoRA/RLHF）、推理优化等核心技术方向。适合 AI 入门者、准备 AI 面试的候选人、以及正在开发 AI 应用的工程师。",

  theme,

  bundler: viteBundler({
    viteOptions: {
      build: {
        rollupOptions: {
          checks: {
            pluginTimings: false,
          },
          onLog(level, log, defaultHandler) {
            if (
              log.code === "INVALID_ANNOTATION" &&
              (log.id?.includes("@vueuse") || log.message?.includes("@vueuse"))
            ) {
              return;
            }
            defaultHandler(level, log);
          },
          onwarn(warning, warn) {
            if (warning.code === "INVALID_ANNOTATION" && warning.message?.includes("@vueuse")) {
              return;
            }
            warn(warning);
          },
        },
      },
    },
  }),

  head: [
    ["link", { rel: "icon", href: "/favicon.png" }],
    ["link", { rel: "apple-touch-icon", href: "/favicon.png" }],
    ["link", { rel: "preload", href: "/logo.png", as: "image", type: "image/png" }],
    [
      "style",
      {},
      [
        "html { background: #f7f9fc; }",
        "body { min-height: 100vh; background: #f7f9fc; }",
        "#app { visibility: hidden; } body.app-ready #app { visibility: visible; }",
        "body::before { content: ''; position: fixed; inset: 0; z-index: 2147483646; opacity: 1; visibility: visible; pointer-events: all; background: radial-gradient(circle at 50% 42%, rgba(9, 109, 217, 0.13), transparent 24rem), linear-gradient(135deg, #f9fcff 0%, #f2f7fd 100%); transition: opacity 0.32s ease, visibility 0.32s ease; }",
        "body::after { content: ''; position: fixed; top: calc(50% - 4rem); left: 50%; z-index: 2147483647; width: min(22rem, 72vw); height: min(8rem, 27vw); transform: translateX(-50%); background: url('/logo.png') center / contain no-repeat; filter: drop-shadow(0 0.9rem 1.5rem rgba(9, 109, 217, 0.18)); animation: anyi-splash-float 1.6s ease-in-out infinite alternate; transition: opacity 0.26s ease, visibility 0.26s ease; }",
        "body.app-ready::before, body.app-ready::after { opacity: 0; visibility: hidden; pointer-events: none; }",
        "@keyframes anyi-splash-float { from { transform: translateX(-50%) translateY(-0.35rem); } to { transform: translateX(-50%) translateY(0.35rem); } }",
        "@media (prefers-reduced-motion: reduce) { body::after { animation: none; } }",
      ].join(""),
    ],
    ["meta", { name: "keywords", content: "AI学习,人工智能,机器学习,深度学习,大模型,LLM,Transformer,RAG,Agent,Prompt,面试题,AI教程,LoRA,RLHF,向量数据库,Embedding,微调,推理优化" }],
    ["meta", { name: "author", content: "安逸Ai" }],
    ["meta", { name: "robots", content: "index, follow, max-snippet:-1, max-image-preview:large" }],
    ["meta", { property: "og:site_name", content: "Anyi AI" }],
    ["meta", { property: "og:locale", content: "zh_CN" }],
    ["meta", { name: "twitter:card", content: "summary_large_image" }],
    ...(googleSiteVerification
      ? [["meta", { name: "google-site-verification", content: googleSiteVerification }] as const]
      : []),
  ],
});
