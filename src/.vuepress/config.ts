import { defineUserConfig } from "vuepress";
import { viteBundler } from "@vuepress/bundler-vite";
import { fileURLToPath } from "node:url";

import theme from "./theme.js";
import { conciseNavigation } from "./plugins/conciseNavigation.js";

const googleSiteVerification = process.env.GOOGLE_SITE_VERIFICATION;

export default defineUserConfig({
  base: "/",

  lang: "zh-CN",
  title: "AI 学习资源站",
  description:
    "Anyi AI 是一个免费开源的 AI 学习资源站，提供系统化的 AI 学习路径。涵盖机器学习基础、深度学习、Transformer 架构、大模型（LLM）、RAG 检索增强生成、Agent 智能体、Prompt Engineering、模型微调（LoRA/RLHF）、推理优化等核心技术方向。适合 AI 入门者、准备 AI 面试的候选人、以及正在开发 AI 应用的工程师。",

  theme,
  alias: {
    "@theme-hope/components/navbar/Navbar": fileURLToPath(new URL("./components/Navbar.vue", import.meta.url)),
    "@theme-hope/components/navbar/NavbarDropdown": fileURLToPath(new URL("./components/NavbarDropdown.vue", import.meta.url)),
    "@theme-hope/components/navbar/NavScreenMenu": fileURLToPath(new URL("./components/NavScreenMenu.vue", import.meta.url)),
  },
  plugins: [conciseNavigation],

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
    ["link", { rel: "preload", href: "/fonts/noto-heading-700.ttf", as: "font", type: "font/ttf", crossorigin: "anonymous" }],
    ["meta", { name: "theme-color", content: "#edf0ee" }],
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
