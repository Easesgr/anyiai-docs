import type { Plugin } from 'vuepress';
import type { SidebarArrayOptions } from 'vuepress-theme-hope';
import { zhSidebar } from '../sidebar/zh.js';

const labels = new Map<string, string>([['/', '首页']]);

function collectLabels(items: SidebarArrayOptions): void {
  for (const item of items) {
    if (typeof item === 'string') continue;
    if (item.link && item.text) {
      labels.set(item.link.replace(/\.md$/, '.html'), item.text);
    }
    if ('children' in item && Array.isArray(item.children)) {
      collectLabels(item.children);
    }
  }
}

for (const items of Object.values(zhSidebar)) {
  if (Array.isArray(items)) collectLabels(items);
}

export const conciseNavigation: Plugin = {
  name: 'anyi-concise-navigation',
  extendsPage(page) {
    const label = labels.get(decodeURI(page.path));
    if (label) page.routeMeta.shortTitle = label;
  },
};
