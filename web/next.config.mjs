/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  distDir: process.env.TAF_NEXT_DIST_DIR || ".next",
};

export default nextConfig;
