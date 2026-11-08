<script setup lang="ts">
import { RouteLink } from 'vuepress/client';
import { learningStages } from '../data/learning';
import ArrowIcon from './ArrowIcon.vue';
import LearningDirections from './LearningDirections.vue';
</script>

<template>
  <section class="learning-roadmap" aria-label="学习路线导航">
    <div class="roadmap-intro"><h2>按目标选路线</h2><p>每条路线列出了前置知识和阅读顺序。</p></div>
    <LearningDirections />
    <div class="roadmap-heading"><h2>五个学习阶段</h2></div>
    <ol class="roadmap-stages">
      <li v-for="stage in learningStages" :key="stage.id">
        <span class="roadmap-terminal">{{ stage.id }}</span>
        <div><h3>{{ stage.label }}</h3><p>{{ stage.outcome }}</p><span class="roadmap-topics">{{ stage.topics.join(' / ') }}</span></div>
        <RouteLink :to="stage.href" class="text-link">进入课程 <ArrowIcon /></RouteLink>
      </li>
    </ol>
    <p class="roadmap-reading-note">下方是各阶段的说明和练习建议。</p>
  </section>
</template>
