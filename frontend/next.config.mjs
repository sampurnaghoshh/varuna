/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_BASE: process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000/api/v1",
    NEXT_PUBLIC_WS_BASE: process.env.NEXT_PUBLIC_WS_BASE ?? "ws://localhost:8000/api/v1/ws",
  },
};

export default nextConfig;
