/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async redirects() {
    return [
      {
        source: "/experiments/new",
        destination: "/experiments",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
