<template>
  <div v-if="config?.locked" class="read-more-lock" ref="lockRef">
    <button v-if="!unlocked" type="button" class="lock-cover" @click="showDialog = true">
      <span class="lock-btn">阅读全文</span>
    </button>

    <!-- Keep the closed dialog out of SSR: body has no teleport target anchors. -->
    <Teleport v-if="showDialog" to="body">
      <div class="dialog-mask" @click.self="showDialog = false" @keydown="handleDialogKey">
        <div ref="dialogRef" class="dialog" role="dialog" aria-modal="true" aria-labelledby="readmore-dialog-title">
          <button class="dialog-close" @click="showDialog = false" aria-label="关闭">
            <svg viewBox="0 0 24 24" width="14" height="14"><path d="M6 6L18 18M18 6L6 18" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/></svg>
          </button>

          <div class="dialog-header">
            <span class="header-icon">
              <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L4 5v6c0 5.5 3.5 10.4 8 11.5 4.5-1.1 8-6 8-11.5V5l-8-3z"/><path d="M9 12l2 2 4-4"/></svg>
            </span>
            <h3 id="readmore-dialog-title">阅读验证</h3>
          </div>

          <p class="dialog-desc">输入公众号验证码，即可继续阅读。一次验证，全站通用。</p>

          <div class="qr-card">
            <div class="qr-wrap">
              <img :src="config.qrcode" alt="公众号二维码" class="qr-img" />
            </div>

            <p class="qr-text">扫码/微信搜索关注 <em>"{{ config.account }}"</em></p>
            <p class="qr-text">回复 <em>"验证码"</em> 获取解锁码</p>
          </div>

          <div class="divider"></div>

          <div class="hint-row">
            <span class="hint">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.1V18h6v-1.2c0-.8.4-1.6 1-2.1A7 7 0 0 0 12 2z"/></svg>
              已关注？直接输入验证码即可
            </span>
            <button class="copy-btn" @click="copyKeyword">
              <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
              {{ copied ? "已复制" : "复制关键词" }}
            </button>
          </div>

          <div class="input-row">
            <input
              v-model="inputCode"
              type="text"
              placeholder="输入验证码"
              aria-label="公众号验证码"
              @keyup.enter="verify"
            />
            <button class="unlock-btn" @click="verify">
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="11" width="16" height="11" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>
              立即解锁
            </button>
          </div>
          <p v-if="errorMsg" class="error-msg" role="alert">{{ errorMsg }}</p>

          <p class="dialog-footer">
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L4 5v6c0 5.5 3.5 10.4 8 11.5 4.5-1.1 8-6 8-11.5V5l-8-3z"/></svg>
            验证码仅用于验证身份，保障您的账号与阅读安全
          </p>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick, watch, computed } from "vue";
import { useRoute } from "vuepress/client";
import readmoreConfig from "../readmore.config.js";

const GLOBAL_KEY = "readmore-verified";

const route = useRoute();
const config = computed(() => {
  const path = route.path;
  return (
    readmoreConfig[path] ||
    readmoreConfig[path.replace(/\/[^/]*\.html$/, "/")] ||
    null
  );
});

const unlocked = ref(false);
const showDialog = ref(false);
const inputCode = ref("");
const errorMsg = ref("");
const copied = ref(false);
const lockRef = ref(null);
const dialogRef = ref(null);
let lastFocused = null;
watch(showDialog, async (open) => {
  if (open) {
    lastFocused = document.activeElement;
    await nextTick();
    dialogRef.value?.querySelector('input')?.focus();
  } else {
    lastFocused?.focus();
  }
});
function handleDialogKey(event) {
  if (event.key === 'Escape') showDialog.value = false;
  if (event.key !== 'Tab') return;
  const controls = [...(dialogRef.value?.querySelectorAll('button, input, a[href]') || [])];
  const first = controls[0];
  const last = controls[controls.length - 1];
  if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
  else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
}
let observer = null;

function setHidden(hidden) {
  const el = lockRef.value;
  if (!el) return;
  // DOM 结构: #markdown-content > [markdown elements directly, no wrapper]
  const mdContent = el.closest("#markdown-content");
  if (!mdContent) return;
  let past = false;
  for (const child of mdContent.children) {
    if (child === el || child.contains(el)) {
      past = true;
      continue;
    }
    if (past) child.style.display = hidden ? "none" : "";
  }
}

function applyLock() {
  if (!config.value || unlocked.value) return;
  const saved = localStorage.getItem(GLOBAL_KEY);
  if (config.value.codeHashes?.includes(saved)) {
    unlocked.value = true;
    return;
  }
  nextTick(() => setHidden(true));
  setTimeout(() => setHidden(true), 50);
  setTimeout(() => setHidden(true), 200);
  setTimeout(() => setHidden(true), 500);
  disconnectObserver();
  const mdContent = lockRef.value?.closest("#markdown-content");
  if (!mdContent) return;
  observer = new MutationObserver(() => {
    if (!unlocked.value) setHidden(true);
  });
  observer.observe(mdContent, { childList: true });
}

function disconnectObserver() {
  if (observer) {
    observer.disconnect();
    observer = null;
  }
}

onMounted(() => {
  applyLock();
});

onUnmounted(() => {
  disconnectObserver();
});

// VuePress 客户端导航时重新应用锁定
watch(
  () => route.path,
  () => {
    if (!unlocked.value) {
      nextTick(() => applyLock());
    }
  },
);

watch(unlocked, (val) => {
  if (val) {
    disconnectObserver();
    setHidden(false);
  }
});

async function verify() {
  errorMsg.value = "";
  if (!inputCode.value.trim()) {
    errorMsg.value = "请输入验证码";
    return;
  }
  const hash = await sha256(inputCode.value.trim());
  if (config.value.codeHashes?.includes(hash)) {
    unlocked.value = true;
    showDialog.value = false;
    localStorage.setItem(GLOBAL_KEY, hash);
  } else {
    errorMsg.value = "验证码错误，请重试";
  }
}

async function copyKeyword() {
  const text = config.value?.keyword;
  if (!text) return;
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    copied.value = true;
    setTimeout(() => (copied.value = false), 1500);
  } catch {
    // ignore
  }
}

async function sha256(message) {
  const msgBuffer = new TextEncoder().encode(message);
  const hashBuffer = await crypto.subtle.digest("SHA-256", msgBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
}
</script>

<style scoped>
.lock-cover { display: block; position: relative; z-index: 20; width: 100%; padding: 5rem 0 2rem; margin-top: -10rem; border: 0; text-align: center; cursor: pointer; background: linear-gradient(to bottom, transparent, var(--site-paper) 60%); }
.lock-btn { display: inline-block; padding: 0.7rem 1.7rem; border: 1px solid var(--site-green); border-radius: 6px; color: var(--site-green-dark); background: var(--site-paper); font-size: 0.9rem; font-weight: 700; }
.lock-cover:hover .lock-btn { color: white; background: var(--site-green); }
.dialog-mask { position: fixed; inset: 0; z-index: 9999; display: flex; align-items: center; justify-content: center; padding: 1rem; background: rgb(23 35 30 / 50%); }
.dialog { position: relative; width: 520px; max-width: 100%; max-height: calc(100dvh - 2rem); overflow-y: auto; padding: 2rem; border-radius: 14px; color: var(--site-ink); background: var(--site-paper); box-shadow: var(--site-elevation); }
.dialog-close { position: absolute; top: 18px; right: 18px; display: grid; place-items: center; width: 36px; height: 36px; border: 0; border-radius: 6px; color: var(--site-muted); background: var(--site-ground); cursor: pointer; }
.dialog-close:hover { color: var(--site-green); background: var(--site-soft); }
.dialog-header { display: flex; align-items: center; gap: 12px; margin-bottom: 1rem; }
.header-icon { display: grid; place-items: center; width: 38px; height: 38px; border-radius: 50%; color: var(--site-green); background: var(--site-soft); }
.dialog-header h3 { margin: 0; color: var(--site-ink); font-size: 1.3rem; font-weight: 700; }
.dialog-desc { margin: 0 0 1.25rem; color: var(--site-muted); font-size: 0.84rem; line-height: 1.8; }
.qr-card { padding: 1.2rem; border-radius: 10px; background: var(--site-soft); text-align: center; }
.qr-wrap { display: inline-block; padding: 8px; border-radius: 5px; background: white; }
.qr-img { display: block; width: 144px; height: 144px; object-fit: contain; }
.qr-text { margin: 0.55rem 0 0; color: #38513f; font-size: 0.83rem; line-height: 1.8; }
.qr-text em { color: var(--site-green-dark); font-style: normal; font-weight: 700; }
.divider { height: 1px; margin-block: 1.25rem; background: var(--site-line); }
.hint-row { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: 0.85rem; }
.hint { display: inline-flex; align-items: center; gap: 6px; color: var(--site-muted); font-size: 0.76rem; }
.hint svg { color: var(--site-green); }
.copy-btn { display: inline-flex; align-items: center; flex: 0 0 auto; gap: 5px; padding: 0.4rem; border: 0; color: var(--site-green-dark); background: transparent; font-size: 0.75rem; cursor: pointer; }
.copy-btn:hover { background: var(--site-soft); }
.input-row { display: flex; align-items: stretch; gap: 0.75rem; }
.input-row input { box-sizing: border-box; flex: 1; min-width: 0; height: 48px; padding: 0 0.85rem; border: 1px solid #9ab2a1; border-radius: 6px; color: var(--site-ink); background: white; font-size: 0.88rem; }
.input-row input::placeholder { color: var(--site-muted); }
.unlock-btn { display: inline-flex; align-items: center; justify-content: center; flex: 1; gap: 6px; min-height: 48px; padding: 0.6rem; border: 0; border-radius: 6px; color: white; background: var(--site-green); font-size: 0.88rem; font-weight: 700; cursor: pointer; }
.unlock-btn:hover { background: var(--site-green-dark); }
.error-msg { margin: 0.75rem 0 0; color: #a53129; font-size: 0.8rem; }
.dialog-footer { display: flex; align-items: center; justify-content: center; gap: 6px; margin: 1rem 0 0; color: var(--site-muted); font-size: 0.66rem; line-height: 1.7; }
.dialog-footer svg { flex: 0 0 auto; }
@media (max-width: 520px) {
  .dialog { padding: 1.4rem; }
  .input-row { flex-direction: column; }
  .input-row input { flex: 0 0 48px; }
  .hint-row { gap: 0.5rem; }
  .hint { font-size: 0.7rem; }
}
</style>
