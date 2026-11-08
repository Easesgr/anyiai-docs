<script setup lang="ts">
import { computed } from 'vue';
import { RouteLink, usePageData } from 'vuepress/client';
import { Layout as HopeLayout } from 'vuepress-theme-hope/client';
import LearningRoadmap from '../components/LearningRoadmap.vue';
import ResourceCTA from '../components/ResourceCTA.vue';
import ArrowIcon from '../components/ArrowIcon.vue';
import ArticleOutline from '../components/ArticleOutline.vue';

const page = usePageData();
const isLearningHub = computed(() => page.value.path === '/learning-path/');
const showResources = computed(() => page.value.path !== '/' && page.value.path !== '/404.html' && page.value.path !== '/about/pdf-download.html');
</script>

<template>
  <HopeLayout>
    <template #toc="headers"><ArticleOutline :headers="headers" /></template>
    <template #contentBefore><LearningRoadmap v-if="isLearningHub" /></template>
    <template #contentAfter><ResourceCTA v-if="showResources" compact /></template>
    <template #navScreenBottom>
      <div class="mobile-nav-resource"><RouteLink to="/about/pdf-download.html">领取 PDF <ArrowIcon /></RouteLink><span>微信公众号 · 安逸Ai</span></div>
    </template>
  </HopeLayout>
</template>
