<script setup lang="ts">
import type { AutoLinkOptions, NavGroup } from 'vuepress-theme-hope/client';
import NavCategory from './NavCategory.vue';

defineProps<{ config: NavGroup<AutoLinkOptions | NavGroup<AutoLinkOptions>> }>();
</script>

<template><NavCategory :config="config"><template #title><slot name="title" /></template></NavCategory></template>
