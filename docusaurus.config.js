// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Automated and Connected Driving',
  tagline: 'This Knowledgebase will challenge your creativity, broaden your technical skills, and inspire you to shape the future of mobility',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://your-docusaurus-site.example.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'facebook', // Usually your GitHub org/user name.
  projectName: 'docusaurus', // Usually your repo name.

  onBrokenLinks: 'throw',
  //onBrokenMarkdownLinks: 'ignore',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.js',
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
          // Add remark-math and rehype-katex plugins
          remarkPlugins: [require('remark-math')],
          rehypePlugins: [require('rehype-katex')],
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],
  

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'ACD',
        /*logo: {
          alt: 'My Site Logo',
          src: 'img/logo.png',
        },*/
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'docsSidebar',
            position: 'left',
            label: 'Theory',
          },
          {
            type: 'docSidebar',
            sidebarId: 'taskSidebar',
            position: 'left',
            label: 'Tasks',
          },
          /*
          {
            type: 'docSidebar',
            sidebarId: 'cppSidebar',
            position: 'left',
            label: 'C++',
          },
          
          {
            type: 'docSidebar',
            sidebarId: 'pySidebar',
            position: 'left',
            label: 'Python',
          },
          {
            type: 'docSidebar',
            sidebarId: 'rosSidebar',
            position: 'left',
            label: 'ROS',
          },
          {
            type: 'docSidebar',
            sidebarId: 'ros2Sidebar',
            position: 'left',
            label: 'ROS2',
          },
          */
          {to: '/blog', label: 'Blog', position: 'left'},
          {
            href: 'https://github.com/facebook/docusaurus',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'ACD',
            items: [
              {
                label: 'Theory',
                to: '/docs/category/before-you-begin',
              },
              {
                label: 'Tasks',
                to: '/docs/category/introduction--tools-1',
              },
            ],
          },
          {
            title: 'Coding',
            items: [
              {
                label: 'C++',
                to: '/docs/category/basics',
              },
              {
                label: 'Python',
                to: '/docs/category/basics-1',
              },
            ],
          },
          {
            title: 'Robot Operating System',
            items: [
              {
                label: 'ROS',
                to: '/docs/category/basics',
              },
              {
                label: 'ROS2',
                to: '/docs/category/basics-1',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'Blog',
                to: '/blog',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/CagriCatik/ACD',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} - Automated and Connected Driving.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
