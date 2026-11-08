<script setup lang="ts">
import { RouteLink } from 'vuepress/client';
import ArrowIcon from './ArrowIcon.vue';

withDefaults(defineProps<{ compact?: boolean }>(), { compact: false });
const resources = ['AI基础', 'AI经典面试题', '大模型100讲', '前沿专题'];
</script>

<template>
  <aside class="resource-cta" :class="{ 'resource-cta-compact': compact }" aria-label="领取学习资料">
    <div class="resource-copy">
      <h2>领取 PDF 资料</h2>
      <p>关注微信公众号「安逸Ai」，回复对应关键词，免费领取四份 PDF 学习资料。</p>
      <div class="resource-keywords"><span v-for="keyword in resources" :key="keyword">{{ keyword }}</span></div>
      <RouteLink to="/about/pdf-download.html" class="text-link">领取方式 <ArrowIcon /></RouteLink>
    </div>
    <figure class="resource-qr">
      <img src="/images/wechat-qrcode.png" width="144" height="144" alt="微信公众号安逸Ai二维码" loading="lazy" />
      <figcaption>微信扫码 · 安逸Ai</figcaption>
    </figure>
  </aside>
</template>
