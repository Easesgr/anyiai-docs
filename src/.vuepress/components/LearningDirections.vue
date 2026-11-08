<script setup lang="ts">
import { computed, ref } from 'vue';
import { RouteLink } from 'vuepress/client';
import { learningDirections } from '../data/learning';
import ArrowIcon from './ArrowIcon.vue';

const selected = ref(0);
const direction = computed(() => learningDirections[selected.value]);
const tabs = ref<HTMLButtonElement[]>([]);
function handleKey(event: KeyboardEvent, index: number) {
  let target = index;
  if (event.key === 'ArrowDown' || event.key === 'ArrowRight') target = (index + 1) % learningDirections.length;
  else if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') target = (index + learningDirections.length - 1) % learningDirections.length;
  else if (event.key === 'Home') target = 0;
  else if (event.key === 'End') target = learningDirections.length - 1;
  else return;
  event.preventDefault();
  selected.value = target;
  tabs.value[target]?.focus();
}
</script>

<template>
  <div class="learning-directions">
    <div class="direction-tabs" role="tablist" aria-label="学习方向">
      <button v-for="(item, index) in learningDirections" :id="`direction-${item.id}`" :key="item.id" ref="tabs" type="button" role="tab" :aria-selected="selected === index" aria-controls="direction-panel" :tabindex="selected === index ? 0 : -1" :class="{ active: selected === index }" @click="selected = index" @keydown="handleKey($event, index)">
        <span><strong>{{ item.label }}</strong><small>{{ item.description }}</small></span><ArrowIcon />
      </button>
    </div>
    <div id="direction-panel" class="direction-panel" role="tabpanel" :aria-labelledby="`direction-${direction.id}`">
      <RouteLink v-for="item in direction.routes" :key="item.href" :to="item.href" class="direction-route">
        <span class="route-level">{{ item.stage }}</span>
        <span class="route-info"><strong>{{ item.title }}</strong><span>{{ item.description }}</span></span>
        <ArrowIcon />
      </RouteLink>
    </div>
  </div>
</template>
