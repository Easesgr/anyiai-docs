import { defineClientConfig } from 'vuepress/client';
import HomePortal from './components/HomePortal.vue';
import ReadMoreLock from './components/ReadMoreLock.vue';
import SiteBrand from './components/SiteBrand.vue';
import ResourceLink from './components/ResourceLink.vue';
import Layout from './layouts/Layout.vue';

export default defineClientConfig({
  layouts: { Layout },
  enhance: ({ app, router }) => {
    app.component('HomePortal', HomePortal);
    app.component('ReadMoreLock', ReadMoreLock);
    app.component('SiteBrand', SiteBrand);
    app.component('ResourceLink', ResourceLink);
    const scrollBehavior = router.options.scrollBehavior;
    router.options.scrollBehavior = async (to, from, savedPosition) => {
      const position = await scrollBehavior?.(to, from, savedPosition);
      if (!to.hash || savedPosition || !position || !('el' in position)) return position;
      const navbarHeight = document.querySelector('#navbar')?.getBoundingClientRect().height ?? 0;
      const outlineHeight = window.matchMedia('(max-width: 1439px)').matches
        ? document.querySelector('.reading-outline > summary')?.getBoundingClientRect().height ?? 0
        : 0;
      return { ...position, top: navbarHeight + outlineHeight + 24 };
    };
  },
});
