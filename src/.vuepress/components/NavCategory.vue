<script setup lang="ts">
import { computed, nextTick, ref, useId, watch } from 'vue';
import { useRoute } from 'vuepress/client';
import AutoLink from 'vuepress-theme-hope/components/base/AutoLink';
import type { AutoLinkOptions, NavGroup } from 'vuepress-theme-hope/client';

const props = withDefaults(defineProps<{
  config: NavGroup<AutoLinkOptions | NavGroup<AutoLinkOptions>>;
  mobile?: boolean;
}>(), { mobile: false });

const route = useRoute();
const menuId = `category-${useId()}`;
const root = ref<HTMLElement>();
const toggle = ref<HTMLButtonElement>();
const open = ref(false);
const active = computed(() => props.config.link && route.path.startsWith(props.config.link));
const categoryLink = computed<AutoLinkOptions | null>(() => props.config.link
  ? { ...props.config, link: props.config.link }
  : null);

function close() {
  open.value = false;
}

function onPointerEnter(event: PointerEvent) {
  if (!props.mobile && event.pointerType === 'mouse') open.value = true;
}

function onPointerLeave() {
  if (!props.mobile && !root.value?.contains(document.activeElement)) close();
}

function onFocusOut(event: FocusEvent) {
  if (!root.value?.contains(event.relatedTarget as Node | null)) close();
}

function onEscape(event: KeyboardEvent) {
  if (!open.value) return;
  event.preventDefault();
  event.stopPropagation();
  close();
  toggle.value?.focus();
}

async function focusFirstLink() {
  open.value = true;
  await nextTick();
  root.value?.querySelector<HTMLAnchorElement>('.nav-category-menu a')?.focus();
}

watch(() => route.fullPath, close);
</script>

<template>
  <div
    ref="root"
    class="nav-category"
    :class="{ 'nav-category-mobile': mobile, 'is-open': open, 'is-active': active }"
    @pointerleave="onPointerLeave"
    @focusout="onFocusOut"
    @keydown.esc="onEscape"
  >
    <div class="nav-category-heading">
      <AutoLink v-if="categoryLink" :config="categoryLink" class="nav-category-link" icon-sizing="height" @click="close" @pointerenter="onPointerEnter" @keydown.down.prevent="focusFirstLink" />
      <span v-else class="nav-category-label" @pointerenter="onPointerEnter"><slot name="title">{{ config.text }}</slot></span>
      <button
        ref="toggle"
        type="button"
        class="nav-category-toggle"
        :aria-label="`${config.text}子目录`"
        :aria-expanded="open"
        :aria-controls="menuId"
        @click="open = !open"
        @keydown.down.prevent="focusFirstLink"
      >
        <svg viewBox="0 0 16 16" width="12" height="12" fill="none" aria-hidden="true"><path d="m4 6 4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" /></svg>
      </button>
    </div>
    <ul v-show="open" :id="menuId" class="nav-category-menu" :aria-label="`${config.text}子目录`" @click="close">
      <li v-for="child in config.children" :key="child.text">
        <template v-if="'children' in child">
          <AutoLink v-if="child.link" :config="{ ...child, link: child.link }" class="nav-category-subtitle" />
          <span v-else class="nav-category-subtitle">{{ child.text }}</span>
          <ul class="nav-category-submenu">
            <li v-for="grandchild in child.children" :key="grandchild.link"><AutoLink :config="grandchild" /></li>
          </ul>
        </template>
        <AutoLink v-else :config="child" />
      </li>
    </ul>
  </div>
</template>
