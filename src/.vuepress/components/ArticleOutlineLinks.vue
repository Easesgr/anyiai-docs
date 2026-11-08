<script setup lang="ts">
import { RouteLink, useRoute } from 'vuepress/client';
import type { PageHeader } from 'vuepress/client';

defineProps<{ headers: PageHeader[] }>();
const emit = defineEmits<{ navigate: [] }>();
const route = useRoute();
</script>

<template>
  <ul class="reading-outline-links">
    <li v-for="header in headers" :key="header.slug">
      <RouteLink :to="`#${header.slug}`" :aria-current="route.hash === `#${header.slug}` ? 'location' : undefined" @click="emit('navigate')">{{ header.title }}</RouteLink>
      <ArticleOutlineLinks v-if="header.children.length" :headers="header.children" @navigate="emit('navigate')" />
    </li>
  </ul>
</template>
