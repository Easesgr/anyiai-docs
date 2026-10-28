import { hopeTheme } from "vuepress-theme-hope";

import { zhNavbar } from "./navbar/index.js";
import { zhSidebar } from "./sidebar/index.js";

export default hopeTheme({
  hostname: "https://www.anyiai.cc",
  titleSeparator: " | ",
  title: "AI学习资源站",

  author: {
    name: "安逸Ai",
  },

  logo: "/logo.png",

  darkmode: "disable",

  repo: "Easesgr/anyiai-docs",

  docsDir: "src",

  navbar: zhNavbar,

  sidebar: zhSidebar,

  headerDepth: 3,

  footer: '<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:0.5rem"><span><a href="https://beian.miit.gov.cn/" target="_blank" rel="noopener noreferrer">赣ICP备2026013262号</a> | 主题: <a href="https://theme-hope.vuejs.press/" target="_blank" rel="noopener noreferrer">VuePress Theme Hope</a></span><span>Copyright © 2026 安逸Ai</span></div>',

  copyright: false,

  displayFooter: true,

  editLink: false,

  lastUpdated: true,

  contributors: true,

  metaLocales: {
    lastUpdated: "最近更新",
    contributors: "贡献者",
  },

  encrypt: {
    config: {
      "/demo/encrypt.html": {
        hint: "Password: 1234",
        password: "1234",
      },
    },
  },

  // These features are enabled for demo, only preserve features you need here
  markdown: {
    align: true,
    attrs: false,
    codeTabs: true,
    component: true,
    demo: true,
    figure: true,
    gfm: true,
    imgLazyload: true,
    imgSize: true,
    include: true,
    mark: true,
    plantuml: true,
    spoiler: true,
    stylize: [
      {
        matcher: "Recommended",
        // oxlint-disable-next-line typescript/consistent-return
        replacer: ({ tag }) => {
          if (tag === "em") {
            return {
              tag: "Badge",
              attrs: { type: "tip" },
              content: "Recommended",
            };
          }
        },
      },
    ],
    sub: true,
    sup: true,
    tabs: true,
    tasklist: true,
    vPre: true,

    // uncomment these if you need TeX support
    // math: {
    //   // install katex before enabling it
    //   type: "katex",
    //   // or install @mathjax/src before enabling it
    //   type: "mathjax",
    // },

    // install chart.js before enabling it
    // chartjs: true,

    // install echarts before enabling it
    // echarts: true,

    // install flowchart.ts before enabling it
    // flowchart: true,

    // install mermaid before enabling it
    // mermaid: true,

    // playground: {
    //   presets: ["ts", "vue"],
    // },

    // install @vue/repl before enabling it
    // vuePlayground: true,

    // install sandpack-vue3 before enabling it
    // sandpack: true,

    // install @vuepress/plugin-revealjs and uncomment these if you need slides
    // revealjs: {
    //   plugins: ["highlight", "math", "search", "notes", "zoom"],
    // },
  },

  plugins: {
    components: {
      components: ["Badge", "VPCard"],
    },

    icon: {
      assets: "iconify",
    },

    slimsearch: true,

    // 为每个页面生成独立的首选地址，避免所有文章被错误归并到首页。
    seo: {
      canonical: (page) => new URL(page.path, "https://www.anyiai.cc").href,
      fallBackImage: "https://www.anyiai.cc/logo.png",
    },

    // 内容站以稳定更新为主，周级频率比无差别 daily 更符合实际。
    sitemap: {
      changefreq: "weekly",
    },

    // Install @vuepress/plugin-pwa and uncomment these if you want a PWA
    // pwa: {
    //   favicon: "/favicon.ico",
    //   cacheHTML: true,
    //   cacheImage: true,
    //   appendBase: true,
    //   apple: {
    //     icon: "/assets/icon/apple-icon-152.png",
    //     statusBarColor: "black",
    //   },
    //   msTile: {
    //     image: "/assets/icon/ms-icon-144.png",
    //     color: "#ffffff",
    //   },
    //   manifest: {
    //     icons: [
    //       {
    //         src: "/assets/icon/chrome-mask-512.png",
    //         sizes: "512x512",
    //         purpose: "maskable",
    //         type: "image/png",
    //       },
    //       {
    //         src: "/assets/icon/chrome-mask-192.png",
    //         sizes: "192x192",
    //         purpose: "maskable",
    //         type: "image/png",
    //       },
    //       {
    //         src: "/assets/icon/chrome-512.png",
    //         sizes: "512x512",
    //         type: "image/png",
    //       },
    //       {
    //         src: "/assets/icon/chrome-192.png",
    //         sizes: "192x192",
    //         type: "image/png",
    //       },
    //     ],
    //     shortcuts: [
    //       {
    //         name: "Demo",
    //         short_name: "Demo",
    //         url: "/demo/",
    //         icons: [
    //           {
    //             src: "/assets/icon/guide-maskable.png",
    //             sizes: "192x192",
    //             purpose: "maskable",
    //             type: "image/png",
    //           },
    //         ],
    //       },
    //     ],
    //   },
    // },
  },
});
