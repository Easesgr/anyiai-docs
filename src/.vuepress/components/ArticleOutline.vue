<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue';
import { useRoute } from 'vuepress/client';
import type { PageHeader } from 'vuepress/client';
import ArticleOutlineLinks from './ArticleOutlineLinks.vue';

defineProps<{ headers: PageHeader[] }>();
const route = useRoute();
const expanded = ref(false);
const details = ref<HTMLDetailsElement>();
let desktop: MediaQueryList | undefined;

function setDefault() {
  expanded.value = desktop?.matches ?? false;
}

function onNavigate() {
  if (!desktop?.matches) expanded.value = false;
}

onMounted(() => {
  desktop = window.matchMedia('(min-width: 1440px)');
  setDefault();
  desktop.addEventListener('change', setDefault);
});
onUnmounted(() => desktop?.removeEventListener('change', setDefault));
watch(() => route.path, setDefault);
</script>

<template>
  <details v-if="headers.length" ref="details" class="reading-outline" :open="expanded" @toggle="expanded = details?.open ?? false">
    <summary>本页目录<svg viewBox="0 0 16 16" width="14" height="14" fill="none" aria-hidden="true"><path d="m4 6 4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" /></svg></summary>
    <nav aria-label="本页目录"><ArticleOutlineLinks :headers="headers" @navigate="onNavigate" /></nav>
  </details>
</template>
