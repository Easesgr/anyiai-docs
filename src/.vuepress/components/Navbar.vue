<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, resolveDynamicComponent, watch } from 'vue';
import type { Component } from 'vue';
import { useRoute } from 'vuepress/client';
import AppearanceButton from 'vuepress-theme-hope/components/appearance/AppearanceButton';
import NavbarBrand from 'vuepress-theme-hope/components/navbar/NavbarBrand';
import NavbarLinks from 'vuepress-theme-hope/components/navbar/NavbarLinks';
import NavScreen from 'vuepress-theme-hope/components/navbar/NavScreen';
import LanguageDropdown from 'vuepress-theme-hope/components/navbar/LanguageDropdown';
import RepoLink from 'vuepress-theme-hope/components/navbar/RepoLink';
import ToggleNavbarButton from 'vuepress-theme-hope/components/navbar/ToggleNavbarButton';
import ToggleSidebarButton from 'vuepress-theme-hope/components/navbar/ToggleSidebarButton';
import { useNavbarAutoHide } from 'vuepress-theme-hope/composables/navbar/useNavbarAutoHide';
import { useThemeLocale } from 'vuepress-theme-hope/composables/useTheme';
import { useWindowSize } from 'vuepress-theme-hope/composables/useWindowSize';
import 'vuepress-theme-hope/styles/navbar/navbar.scss';

const emit = defineEmits<{ toggleSidebar: [] }>();
const route = useRoute();
const theme = useThemeLocale();
const { isMobile } = useWindowSize();
const autoHide = useNavbarAutoHide();
const showScreen = ref(false);
const layout = computed(() => theme.value.navbarLayout ?? {
  start: ['Brand'], center: ['Links'], end: ['Search'],
});
const components: Record<string, Component> = {
  Brand: NavbarBrand, Links: NavbarLinks, Language: LanguageDropdown,
  Repo: RepoLink, Outlook: AppearanceButton,
};
function navbarComponent(name: string) {
  return components[name] ?? resolveDynamicComponent(name === 'Search' ? 'SearchBox' : name);
}

function closeScreen() {
  showScreen.value = false;
}

function toggleSidebar() {
  closeScreen();
  emit('toggleSidebar');
}

function onEscape(event: KeyboardEvent) {
  if (event.key !== 'Escape' || !showScreen.value) return;
  event.preventDefault();
  closeScreen();
  document.querySelector<HTMLButtonElement>('.vp-toggle-navbar-button')?.focus();
}

// Content can mount or update after navigation has finished. Only a new route
// should dismiss a menu the reader has opened in the meantime.
watch(() => route.fullPath, closeScreen);
watch(isMobile, mobile => { if (!mobile) closeScreen(); });
onMounted(() => window.addEventListener('keydown', onEscape));
onUnmounted(() => window.removeEventListener('keydown', onEscape));
</script>

<template>
  <header id="navbar" class="vp-navbar" :class="{ 'auto-hide': autoHide }" vp-navbar>
    <div class="vp-navbar-start">
      <ToggleSidebarButton @toggle="toggleSidebar" />
      <component :is="navbarComponent(name)" v-for="name in layout.start" :key="name" />
    </div>
    <div class="vp-navbar-center"><component :is="navbarComponent(name)" v-for="name in layout.center" :key="name" /></div>
    <div class="vp-navbar-end">
      <component :is="navbarComponent(name)" v-for="name in layout.end" :key="name" />
      <ToggleNavbarButton :active="showScreen" @toggle="showScreen = !showScreen" />
    </div>
  </header>
  <NavScreen :show="showScreen">
    <template #navScreenTop><slot name="navScreenTop" /></template>
    <template #navScreenBottom><slot name="navScreenBottom" /></template>
  </NavScreen>
</template>
