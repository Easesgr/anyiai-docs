<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { RouteLink } from 'vuepress/client';
import { learningStages } from '../data/learning';
import ArrowIcon from './ArrowIcon.vue';
import ResourceCTA from './ResourceCTA.vue';

const preview = (id: string) => `/images/home-previews/${id}.webp`;
const selected = ref(2);
const stage = computed(() => learningStages[selected.value]);
const stageCover = computed(() => preview(stage.value.cover.split('/')[3]));
const stageTabs = ref<HTMLButtonElement[]>([]);
const imageFailed = ref(false);
watch(selected, () => { imageFailed.value = false; });

const stageCarousel = ref<HTMLElement | null>(null);
const carouselAutoplay = ref(false);
const carouselVisible = ref(false);
const reducedMotion = ref(true);
const pageVisible = ref(true);
const carouselDelay = 6500;
const carouselCanAdvance = computed(() => carouselAutoplay.value && carouselVisible.value && pageVisible.value && !reducedMotion.value);
let carouselTimer: ReturnType<typeof setTimeout> | undefined;
let motionPreference: MediaQueryList | undefined;

function selectStage(index: number) {
  selected.value = (index + learningStages.length) % learningStages.length;
  restartCarousel();
}

function handleCarouselFocus(event: FocusEvent) {
  if (event.target instanceof Element && event.target.matches(':focus-visible') && !event.target.closest('.carousel-play')) {
    carouselAutoplay.value = false;
  }
}

function updateMotionPreference() {
  reducedMotion.value = motionPreference?.matches ?? true;
  if (reducedMotion.value) carouselAutoplay.value = false;
}

function toggleCarousel() {
  if (!reducedMotion.value) carouselAutoplay.value = !carouselAutoplay.value;
}

function restartCarousel() {
  clearTimeout(carouselTimer);
  if (carouselCanAdvance.value) {
    carouselTimer = setTimeout(() => { selected.value = (selected.value + 1) % learningStages.length; }, carouselDelay);
  }
}

watch([carouselCanAdvance, selected], restartCarousel);

function handleStageKey(event: KeyboardEvent, index: number) {
  let target = index;
  if (event.key === 'ArrowRight') target = (index + 1) % learningStages.length;
  else if (event.key === 'ArrowLeft') target = (index + learningStages.length - 1) % learningStages.length;
  else if (event.key === 'Home') target = 0;
  else if (event.key === 'End') target = learningStages.length - 1;
  else return;
  event.preventDefault();
  carouselAutoplay.value = false;
  selectStage(target);
  stageTabs.value[target]?.focus();
}

const illustratedArticles = [
  { title: 'RAG 知识问答', id: '1ba02e09ddd23a13', href: '/llm-100/rag/what-is-rag.html' },
  { title: 'Agent 状态机', id: '9cefdf4c75cd9cca', href: '/llm-100/agent-engineering/agent-state-machine.html' },
  { title: 'Transformer', id: '914738cd43afa552', href: '/fundamentals/transformer-arch/' },
  { title: '模型网关', id: 'cba0a54969a72542', href: '/architecture/model-gateway.html' },
  { title: '应用评测', id: 'd9846430d8cb799a', href: '/architecture/ai-evaluation.html' },
  { title: '流式响应', id: 'b271d5f10e97a7de', href: '/architecture/streaming.html' },
];

const collections = [
  { title: 'AI 基础', href: '/fundamentals/', count: '10 个专题', covers: ['04601419fb1c543d', '75675110a67e1d00', '914738cd43afa552'] },
  { title: '大模型 100 讲', href: '/llm-100/', count: '10 个专题', covers: ['02bb1649a60ce290', '5c8e357c3e0455b6', '2c0807554e9880c5'] },
  { title: '前沿专题', href: '/frontier/', count: '15 个专题', covers: ['bbd28bff6f6b5d0a', '298c02597362bdac', '2005fb154a6db027'] },
  { title: '工程架构', href: '/architecture/', count: '12 篇文章', covers: ['cba0a54969a72542', 'b271d5f10e97a7de', '418b800c7970cb70'] },
  { title: '项目实践', href: '/projects/', count: '14 篇文章', covers: ['832b3585039d754b', '0a308926bd0f8d00', '53e33902a9e74d13'] },
  { title: '面试题', href: '/interview/', count: '8 个专题', covers: ['82843f8368aa735d', '16994c1888f90e8c', '26f3d5dcc423189b'] },
];

const heroCollections = [collections[0], collections[4], collections[5]];

const quickRoutes = [
  { title: '开发入门', href: '/learning-path/ai-engineer.html' },
  { title: 'RAG 实战', href: '/learning-path/rag-practice.html' },
  { title: 'Agent 开发', href: '/learning-path/agent-engineer.html' },
  { title: '面试准备', href: '/learning-path/ai-interview-prep.html' },
];

const imageRail = ref<HTMLElement | null>(null);
const railPaused = ref(false);
const railVisible = ref(false);
let visibilityObserver: IntersectionObserver | undefined;

function updateVisibility() {
  pageVisible.value = !document.hidden;
}

onMounted(() => {
  updateVisibility();
  motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
  updateMotionPreference();
  carouselAutoplay.value = !reducedMotion.value;
  motionPreference.addEventListener('change', updateMotionPreference);
  visibilityObserver = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (entry.target === imageRail.value) railVisible.value = entry.isIntersecting;
      if (entry.target === stageCarousel.value) carouselVisible.value = entry.isIntersecting;
    }
  });
  if (imageRail.value) visibilityObserver.observe(imageRail.value);
  if (stageCarousel.value) visibilityObserver.observe(stageCarousel.value);
  document.addEventListener('visibilitychange', updateVisibility);
});

onBeforeUnmount(() => {
  clearTimeout(carouselTimer);
  visibilityObserver?.disconnect();
  motionPreference?.removeEventListener('change', updateMotionPreference);
  document.removeEventListener('visibilitychange', updateVisibility);
});
</script>

<template>
  <div class="course-home">
    <section class="course-hero" aria-labelledby="course-home-title">
      <div class="hero-copy">
        <h1 id="course-home-title"><span>大模型原理</span><span>与应用开发</span></h1>
        <p class="hero-description">RAG、Agent、训练与部署。</p>
        <div class="hero-actions">
          <RouteLink to="/learning-path/" class="button-primary">开始学习 <ArrowIcon /></RouteLink>
          <a href="#course-library" class="button-secondary">浏览课程 <ArrowIcon /></a>
        </div>
        <p class="hero-library-note">502 篇文章<span aria-hidden="true">/</span>43 个专题</p>
        <nav class="hero-course-shelf" aria-label="常用课程">
          <RouteLink v-for="collection in heroCollections" :key="collection.href" :to="collection.href" class="hero-course-entry">
            <img :src="preview(collection.covers[2])" :alt="`${collection.title}课程封面`" width="640" height="400" decoding="async" />
            <span>{{ collection.title }} <ArrowIcon /></span>
          </RouteLink>
        </nav>
      </div>
      <div ref="stageCarousel" class="course-patch-panel" role="region" aria-roledescription="轮播" aria-label="图解课程" @focusin="handleCarouselFocus">
        <div class="stage-tabs" role="tablist" aria-label="五阶段学习路线">
          <button v-for="(item, index) in learningStages" :id="`stage-${item.id}`" :key="item.id" ref="stageTabs" type="button" role="tab" :aria-selected="selected === index" :tabindex="selected === index ? 0 : -1" aria-controls="stage-preview" :class="{ active: selected === index, connected: index <= selected }" @click="selectStage(index)" @keydown="handleStageKey($event, index)">
            <span class="stage-node">{{ item.id }}</span><span class="stage-label">{{ item.label }}</span>
          </button>
        </div>
        <div id="stage-preview" class="stage-preview" role="tabpanel" :aria-labelledby="`stage-${stage.id}`" :aria-live="carouselAutoplay ? 'off' : 'polite'" tabindex="0">
          <Transition name="course-switch" mode="out-in" @before-leave="element => element.setAttribute('inert', '')">
            <div :key="stage.id" class="stage-preview-content">
              <div class="stage-media">
                <img v-if="!imageFailed" :src="stageCover" :alt="`${stage.title}的彩色马克笔讲解图`" width="900" height="600" fetchpriority="high" @error="imageFailed = true" />
                <p v-else class="stage-image-error">讲解图暂时无法加载，可直接进入课程阅读。</p>
              </div>
              <div class="stage-course-copy"><h2>{{ stage.title }}</h2><RouteLink :to="stage.href" class="text-link">阅读课程 <ArrowIcon /></RouteLink></div>
            </div>
          </Transition>
        </div>
        <div class="stage-carousel-controls">
          <span class="carousel-page" aria-hidden="true"><strong>{{ String(selected + 1).padStart(2, '0') }}</strong><span>/ {{ String(learningStages.length).padStart(2, '0') }}</span></span>
          <div class="carousel-buttons">
            <button type="button" class="carousel-previous" aria-label="上一张课程图" aria-controls="stage-preview" @click="selectStage(selected - 1)"><ArrowIcon /></button>
            <button type="button" class="carousel-play" :disabled="reducedMotion" :aria-pressed="carouselAutoplay" aria-controls="stage-preview" @click="toggleCarousel">
              <svg viewBox="0 0 20 20" width="16" height="16" fill="none" aria-hidden="true"><path v-if="!carouselAutoplay" d="m7 4 8 6-8 6V4Z" fill="currentColor" /><path v-else d="M7 4v12m6-12v12" stroke="currentColor" stroke-width="2" stroke-linecap="round" /></svg>
              {{ reducedMotion ? '手动浏览' : carouselAutoplay ? '暂停轮播' : '自动轮播' }}
            </button>
            <button type="button" class="carousel-next" aria-label="下一张课程图" aria-controls="stage-preview" @click="selectStage(selected + 1)"><ArrowIcon /></button>
          </div>
        </div>
      </div>
    </section>

    <nav class="home-quick-routes" aria-label="推荐学习路线">
      <RouteLink to="/learning-path/" class="quick-routes-title">学习路线 <ArrowIcon /></RouteLink>
      <RouteLink v-for="route in quickRoutes" :key="route.href" :to="route.href">{{ route.title }}</RouteLink>
    </nav>

    <section ref="imageRail" class="home-image-gallery home-section" :class="{ 'is-paused': railPaused || !railVisible || !pageVisible }" aria-labelledby="home-gallery-title">
      <div class="section-heading">
        <h2 id="home-gallery-title">图解精选</h2>
        <button type="button" class="rail-toggle" :aria-pressed="railPaused" aria-controls="home-image-rail" @click="railPaused = !railPaused">
          <svg viewBox="0 0 20 20" width="16" height="16" fill="none" aria-hidden="true"><path v-if="railPaused" d="m7 4 8 6-8 6V4Z" fill="currentColor" /><path v-else d="M7 4v12m6-12v12" stroke="currentColor" stroke-width="2" stroke-linecap="round" /></svg>
          {{ railPaused ? '继续轮播' : '暂停轮播' }}
        </button>
      </div>
      <div id="home-image-rail" class="image-rail-viewport">
        <div class="image-rail-track">
          <div v-for="copy in 2" :key="copy" class="image-rail-group" :class="{ 'rail-copy': copy === 2 }" :aria-hidden="copy === 2 ? true : undefined">
            <RouteLink v-for="article in illustratedArticles" :key="article.href" :to="article.href" class="image-rail-item" :tabindex="copy === 2 ? -1 : undefined">
              <img :src="preview(article.id)" :alt="`${article.title}图解`" width="640" height="400" loading="lazy" decoding="async" />
              <span>{{ article.title }} <ArrowIcon /></span>
            </RouteLink>
          </div>
        </div>
      </div>
    </section>

    <section id="course-library" class="home-library home-section" aria-labelledby="course-library-title">
      <div class="section-heading"><h2 id="course-library-title">课程目录</h2><RouteLink to="/resources/" class="text-link">工具与资源 <ArrowIcon /></RouteLink></div>
      <div class="course-folder-grid">
        <RouteLink v-for="collection in collections" :key="collection.href" :to="collection.href" class="course-folder" :aria-label="`${collection.title}，${collection.count}`">
          <div class="course-folder-shell" aria-hidden="true">
            <div class="course-folder-back"></div>
            <div class="course-folder-papers">
              <figure v-for="cover in collection.covers" :key="cover"><img :src="preview(cover)" alt="" width="640" height="400" loading="lazy" decoding="async" /></figure>
            </div>
            <div class="course-folder-front"><strong>{{ collection.title }}</strong><span>{{ collection.count }}</span><ArrowIcon /></div>
          </div>
        </RouteLink>
      </div>
    </section>

    <ResourceCTA />
  </div>
</template>
