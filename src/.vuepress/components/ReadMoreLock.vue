<template>
  <div v-if="config?.locked" class="read-more-lock" ref="lockRef">
    <div v-if="!unlocked" class="lock-cover" @click="showDialog = true">
      <span class="lock-btn">阅读全文</span>
    </div>

    <Teleport to="body">
      <div v-if="showDialog" class="dialog-mask" @click.self="showDialog = false">
        <div class="dialog">
          <button class="dialog-close" @click="showDialog = false" aria-label="关闭">
            <svg viewBox="0 0 24 24" width="14" height="14"><path d="M6 6L18 18M18 6L6 18" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/></svg>
          </button>

          <div class="dialog-header">
            <span class="header-icon">
              <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L4 5v6c0 5.5 3.5 10.4 8 11.5 4.5-1.1 8-6 8-11.5V5l-8-3z"/><path d="M9 12l2 2 4-4"/></svg>
            </span>
            <h3>人机验证</h3>
          </div>

          <p class="dialog-desc">为保障正常阅读体验，本站部分内容已开启一次性验证。验证后全站解锁。</p>

          <div class="qr-card">
            <svg class="deco deco-scan" viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 10V4h6M28 10V4h-6M4 22v6h6M28 22v6h-6"/></svg>
            <svg class="deco deco-chat" viewBox="0 0 40 40" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M5 18a10 10 0 1 1 5 8.7L5 28l1.3-4A10 10 0 0 1 5 18z"/><path d="M20 26a8 8 0 1 0-4-6.9L15 22l3-.8A8 8 0 0 0 20 26z"/></svg>
            <svg class="deco deco-spark s1" viewBox="0 0 24 24" fill="currentColor"><path d="M12 1l2 9 9 2-9 2-2 9-2-9-9-2 9-2z"/></svg>
            <svg class="deco deco-spark s2" viewBox="0 0 24 24" fill="currentColor"><path d="M12 1l2 9 9 2-9 2-2 9-2-9-9-2 9-2z"/></svg>
            <svg class="deco deco-spark s3" viewBox="0 0 24 24" fill="currentColor"><path d="M12 1l2 9 9 2-9 2-2 9-2-9-9-2 9-2z"/></svg>
            <svg class="deco deco-spark s4" viewBox="0 0 24 24" fill="currentColor"><path d="M12 1l2 9 9 2-9 2-2 9-2-9-9-2 9-2z"/></svg>

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
              @keyup.enter="verify"
            />
            <button class="unlock-btn" @click="verify">
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="11" width="16" height="11" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>
              立即解锁
            </button>
          </div>
          <p v-if="errorMsg" class="error-msg">{{ errorMsg }}</p>

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
.lock-cover {
  position: relative;
  z-index: 20;
  padding: 5rem 0 2rem;
  text-align: center;
  cursor: pointer;
  margin-top: -10rem;
  background: linear-gradient(
    to bottom,
    transparent 0%,
    var(--vp-c-bg) 60%,
    var(--vp-c-bg) 100%
  );
}

.lock-btn {
  display: inline-block;
  padding: 0.5rem 1.5rem;
  font-size: 0.9rem;
  font-weight: 500;
  color: #07c160;
  background: transparent;
  border: 1px solid #07c160;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.25s;
}

.lock-btn:hover {
  background: #07c160;
  color: #fff;
}

.dialog-mask {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.55);
  backdrop-filter: blur(2px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
  padding: 1rem;
}

.dialog {
  position: relative;
  background: var(--vp-c-bg, #fff);
  border-radius: 20px;
  width: 560px;
  max-width: 100%;
  padding: 28px 32px 22px;
  box-shadow: 0 24px 60px rgba(2, 32, 18, 0.18),
    0 4px 12px rgba(2, 32, 18, 0.06);
}

.dialog-close {
  position: absolute;
  top: 18px;
  right: 18px;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  border: none;
  background: #f1f3f5;
  color: #6b7280;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: background 0.2s, color 0.2s;
}

.dialog-close:hover {
  background: #e5e7eb;
  color: #111827;
}

.dialog-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
}

.header-icon {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: #e7f8ee;
  color: #07c160;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.dialog-header h3 {
  margin: 0;
  font-size: 22px;
  font-weight: 700;
  color: var(--vp-c-text-1, #111827);
  letter-spacing: 0.5px;
}

.dialog-desc {
  margin: 0 0 18px;
  padding-left: 52px;
  font-size: 13.5px;
  color: var(--vp-c-text-2, #6b7280);
  line-height: 1.6;
}

.qr-card {
  position: relative;
  border-radius: 16px;
  padding: 22px 16px 18px;
  background: linear-gradient(180deg, #f3fbf6 0%, #ecf9f1 100%);
  border: 1px solid rgba(7, 193, 96, 0.22);
  text-align: center;
  overflow: hidden;
}

.deco {
  position: absolute;
  color: #07c160;
  pointer-events: none;
}

.deco-scan {
  width: 42px;
  height: 42px;
  top: 50%;
  left: 28px;
  transform: translateY(-50%);
  opacity: 0.35;
}

.deco-chat {
  width: 56px;
  height: 56px;
  top: 50%;
  right: 26px;
  transform: translateY(-50%);
  opacity: 0.28;
}

.deco-spark {
  width: 14px;
  height: 14px;
  opacity: 0.5;
}

.deco-spark.s1 {
  top: 18px;
  right: 32%;
}
.deco-spark.s2 {
  bottom: 28%;
  left: 24%;
  width: 11px;
  height: 11px;
  opacity: 0.4;
}
.deco-spark.s3 {
  top: 38%;
  right: 14%;
  width: 10px;
  height: 10px;
  opacity: 0.45;
}
.deco-spark.s4 {
  bottom: 22px;
  right: 30%;
  width: 12px;
  height: 12px;
  opacity: 0.45;
}

.qr-wrap {
  position: relative;
  display: inline-block;
  padding: 10px;
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 4px 16px rgba(7, 193, 96, 0.1);
}

.qr-img {
  display: block;
  width: 160px;
  height: 160px;
  border-radius: 6px;
  object-fit: cover;
}

.qr-text {
  margin: 10px 0 0;
  font-size: 14px;
  color: var(--vp-c-text-1, #1f2937);
  line-height: 1.8;
}

.qr-text em {
  font-style: normal;
  color: #07c160;
  font-weight: 600;
}

.divider {
  height: 1px;
  margin: 18px 0 14px;
  background-image: linear-gradient(
    to right,
    #d1d5db 50%,
    transparent 50%
  );
  background-size: 8px 1px;
  background-repeat: repeat-x;
  background-position: left center;
}

.hint-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
  font-size: 13px;
}

.hint {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--vp-c-text-2, #6b7280);
}

.hint svg {
  color: #07c160;
}

.copy-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: none;
  border: none;
  padding: 4px 6px;
  color: #07c160;
  font-size: 13px;
  cursor: pointer;
  border-radius: 6px;
  transition: background 0.2s;
}

.copy-btn:hover {
  background: rgba(7, 193, 96, 0.08);
}

.input-row {
  display: flex;
  gap: 12px;
  align-items: stretch;
}

.input-row input {
  flex: 1;
  min-width: 0;
  padding: 0 16px;
  height: 50px;
  border: 1px solid var(--vp-c-divider, #e5e7eb);
  border-radius: 12px;
  font-size: 15px;
  background: var(--vp-c-bg, #fff);
  color: var(--vp-c-text-1, #1f2937);
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.input-row input::placeholder {
  color: #9ca3af;
}

.input-row input:focus {
  border-color: #07c160;
  box-shadow: 0 0 0 3px rgba(7, 193, 96, 0.15);
}

.unlock-btn {
  flex: 1.4;
  height: 50px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  background: linear-gradient(180deg, #18b863 0%, #07c160 100%);
  color: #fff;
  border: none;
  border-radius: 12px;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  box-shadow: 0 6px 16px rgba(7, 193, 96, 0.28);
  transition: transform 0.15s, box-shadow 0.2s, filter 0.2s;
}

.unlock-btn:hover {
  filter: brightness(1.04);
  box-shadow: 0 8px 20px rgba(7, 193, 96, 0.36);
}

.unlock-btn:active {
  transform: translateY(1px);
}

.error-msg {
  margin: 10px 0 0;
  color: #e74c3c;
  font-size: 13px;
  text-align: center;
}

.dialog-footer {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  margin: 16px 0 0;
  font-size: 12px;
  color: var(--vp-c-text-3, #9ca3af);
}

.dialog-footer svg {
  color: #9ca3af;
}

@media (max-width: 520px) {
  .dialog {
    padding: 22px 18px 18px;
    border-radius: 16px;
  }
  .dialog-header h3 {
    font-size: 19px;
  }
  .dialog-desc {
    padding-left: 0;
  }
  .qr-img {
    width: 140px;
    height: 140px;
  }
  .deco-scan,
  .deco-chat {
    display: none;
  }
  .input-row {
    flex-direction: column;
  }
  .unlock-btn {
    flex: none;
  }
}
</style>
