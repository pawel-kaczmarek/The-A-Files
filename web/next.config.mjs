/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  distDir: process.env.TAF_NEXT_DIST_DIR || ".next",
  // Self-contained server for the Docker image (web/Dockerfile); `next start` does not support it.
  output: process.env.TAF_NEXT_OUTPUT === "standalone" ? "standalone" : undefined,
};

export default nextConfig;
