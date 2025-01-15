// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import { themes as prismThemes } from "prism-react-renderer";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "Automated and Connected Driving",
  tagline:
    "This Knowledgebase will challenge your creativity, broaden your technical skills, and inspire you to shape the future of mobility",
  favicon: "img/favicon.ico",

  // Set the production url of your site here
  url: "https://cagricatik.github.io", // Correct: Root domain without sub-path

  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/Autonomous-Connected-Driving/", // Correct: Sub-path with trailing slash

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "cagricatik", // Usually your GitHub org/user name.
  projectName: "Autonomous-Connected-Driving", // Usually your repo name.

  // Enable GitHub Pages deployment
  deploymentBranch: "gh-pages", // Default is 'gh-pages'
  trailingSlash: false, // Optional: depends on your preference

  // Additional configurations can go here
  // e.g., themeConfig, presets, plugins, etc.

  onBrokenLinks: "ignore",
  //onBrokenMarkdownLinks: 'ignore',
  onBrokenMarkdownLinks: "warn",

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.js",
          editUrl:
          "https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/",
          remarkPlugins: [require("remark-math")],
          rehypePlugins: [require("rehype-katex")],
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ["rss", "atom"],
            xslt: true,
          },
          editUrl:
            "https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/",
          onInlineTags: "warn",
          onInlineAuthors: "warn",
          onUntruncatedBlogPosts: "warn",
        },
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      },
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: "img/docusaurus-social-card.jpg",
      navbar: {
        title: "",
        logo: {
          alt: "My Site Logo",
          src: "img/logo.png",
        },
        items: [
          {
            type: "docSidebar",
            sidebarId: "introToolsSidebar",
            position: "left",
            label: "Introduction & Tools",
          },
          {
            type: "docSidebar",
            sidebarId: "sensorSidebar",
            position: "left",
            label: "Sensor Data Processing",
          },
          {
            type: "docSidebar",
            sidebarId: "objectSidebar",
            position: "left",
            label: "Object Fusion and Tracking",
          },
          {
            type: "docSidebar",
            sidebarId: "vehicleSidebar",
            position: "left",
            label: "Vehicle Guidance",
          },
          {
            type: "docSidebar",
            sidebarId: "connectedSidebar",
            position: "left",
            label: "Connected Driving",
          },
          {
            type: "docSidebar",
            sidebarId: "taskSidebar",
            position: "left",
            label: "Tasks",
          },
          {
            type: "docSidebar",
            sidebarId: "cppSidebar",
            position: "right",
            label: "C++",
          },
          {
            type: "docSidebar",
            sidebarId: "pySidebar",
            position: "right",
            label: "Python",
          },
          {
            type: "docSidebar",
            sidebarId: "rosSidebar",
            position: "right",
            label: "ROS",
          },
          {
            type: "docSidebar",
            sidebarId: "ros2Sidebar",
            position: "right",
            label: "ROS2",
          },
        ],
      },
      footer: {
        style: "dark",
        links: [
          {
            title: "Coding",
            items: [
              {
                label: "C++",
                to: "/docs/cpp/getting_started",
              },
              {
                label: "Python",
                to: "/docs/python/getting_started",
              },
            ],
          },

          {
            title: "Robot Operating System",
            items: [
              {
                label: "ROS",
                to: "/docs/ros/getting_started",
              },
              {
                label: "ROS2",
                to: "/docs/ros2/getting_started",
              },
            ],
          },
          {
            title: "More",
            items: [
              {
                label: "Blog",
                to: "/blog",
              },
              {
                label: "GitHub",
                href: "https://github.com/CagriCatik/Autonomous-Connected-Driving",
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
