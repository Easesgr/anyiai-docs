import { defineClientConfig } from "vuepress/client";
import HomePortal from "./components/HomePortal.vue";
import ReadMoreLock from "./components/ReadMoreLock.vue";

const sectionLinks: Record<string, string> = {
  "AI 基础": "/fundamentals/",
  "AI经典面试题": "/interview/",
  "大模型100讲": "/llm-100/",
  "学习路线": "/learning-path/",
  "前沿专题": "/frontier/",
  "资源导航": "/resources/",
  "关于": "/about/",
};

function bindNavClick() {
  document
    .querySelectorAll<HTMLElement>(".vp-dropdown-title")
    .forEach((title) => {
      if (title.dataset.navBound) return;
      const text = title.getAttribute("aria-label")?.trim();
      const target = text ? sectionLinks[text] : undefined;
      if (!target) return;
      title.dataset.navBound = "1";
      title.style.cursor = "pointer";
      title.addEventListener("click", (e) => {
        if ((e.target as HTMLElement).closest(".vp-dropdown")) return;
        e.preventDefault();
        window.location.href = target;
      });
    });
}

function bindContactModal() {
  const overlay = document.querySelector<HTMLElement>('[data-contact-overlay]')
  if (!overlay) return
  if (overlay.dataset.bound) return
  overlay.dataset.bound = '1'

  // 把模态框移到 body 下，避免父级 transform 影响 fixed 定位
  document.body.appendChild(overlay)

  overlay.addEventListener('click', (e) => {
    const target = e.target as HTMLElement
    if (target === overlay) {
      overlay.classList.remove('active')
      return
    }
    if (target.closest('[data-contact-close]')) {
      overlay.classList.remove('active')
      return
    }
    const copyBtn = target.closest('[data-copy]')
    if (copyBtn) {
      const text = copyBtn.getAttribute('data-copy')
      if (!text) return
      navigator.clipboard.writeText(text).then(() => {
        const label = copyBtn.querySelector('.copy-text')
        if (label) {
          label.textContent = '已复制'
          setTimeout(() => { label.textContent = '复制' }, 1500)
        }
      })
    }
  })

  document.addEventListener('click', (e) => {
    const target = e.target as HTMLElement
    if (target.closest('[data-contact-trigger]')) {
      overlay.classList.add('active')
    }
  })
}

function showPageLoader() {
  document.documentElement.classList.add("route-loading");
  document.body.classList.remove("app-ready");
}

function hidePageLoader() {
  window.setTimeout(() => {
    document.body.classList.add("app-ready");
    document.documentElement.classList.remove("route-loading");
  }, 280);
}

function bindHomeLogoLoader() {
  if (document.documentElement.dataset.homeLogoLoaderBound) return;
  document.documentElement.dataset.homeLogoLoaderBound = "1";

  // 主题的 Logo 图片位于链接内部。捕获阶段先显示占位层，
  // 即便浏览器将这次跳转处理为整页导航，也不会露出白屏。
  document.addEventListener("click", (event) => {
    const target = event.target as HTMLElement;
    const logo = target.closest<HTMLElement>(".vp-nav-logo");
    if (!logo) return;

    const link = logo.closest<HTMLAnchorElement>("a");
    if (link) {
      const destination = new URL(link.href, window.location.href);
      const normalizePath = (path: string) => path.replace(/\/+$/, "") || "/";

      // 已在首页时点击 Logo 不重复导航，也不展示加载占位层。
      if (
        destination.origin === window.location.origin &&
        normalizePath(destination.pathname) === normalizePath(window.location.pathname)
      ) {
        event.preventDefault();
        event.stopImmediatePropagation();
        return;
      }
    }

    showPageLoader();
  }, { capture: true });
}

function init() {
  bindNavClick();
  bindContactModal();
  bindHomeLogoLoader();
  document.body.classList.add('app-ready');
  // 调试阶段暂时关闭复制保护，上线前恢复
  // document.addEventListener("contextmenu", (e) => e.preventDefault());
  // document.addEventListener("copy", (e) => e.preventDefault());
  // document.addEventListener("cut", (e) => e.preventDefault());
  // document.addEventListener("dragstart", (e) => {
  //   if ((e.target as HTMLElement).tagName === "IMG") e.preventDefault();
  // });
}

if (typeof window !== "undefined") {
  if (document.readyState === "complete") init();
  else window.addEventListener("load", init, { once: true });
  // 兜底：3 秒后强制隐藏加载动画
  setTimeout(() => document.body.classList.add('app-ready'), 3000);
  // 延迟轮询，确保 navbar 异步渲染后能绑定
  let attempts = 0;
  const timer = setInterval(() => {
    bindNavClick();
    bindContactModal();
    if (++attempts >= 10) clearInterval(timer);
  }, 300);
}

export default defineClientConfig({
  enhance: ({ app, router }) => {
    app.component("HomePortal", HomePortal);
    app.component("ReadMoreLock", ReadMoreLock);

    if (typeof window !== "undefined") {
      router.beforeEach((to, from) => {
        // 文章滚动时主题可能同步更新 hash。锚点变化不属于页面跳转，
        // 不能重新显示全屏 Logo，否则上下阅读会产生反复加载的错觉。
        if (to.path !== from.path) showPageLoader();
      });

      router.afterEach(() => {
        hidePageLoader();
      });
    }
  },
});
