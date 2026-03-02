import type { NextConfig } from "next";

const config: NextConfig = {
  output: "standalone",

  // Proxy /api/v1/* and /api/v3/* to FastAPI backend
  // Set API_BACKEND_URL in Railway to the API service internal URL
  // e.g. API_BACKEND_URL=http://api.railway.internal:8000
  async rewrites() {
    const backend = process.env.API_BACKEND_URL;
    if (backend) {
      return [
        {
          source: "/api/v1/:path*",
          destination: `${backend}/api/v1/:path*`,
        },
        {
          source: "/api/v3/:path*",
          destination: `${backend}/api/v3/:path*`,
        },
      ];
    }
    return [];
  },
};

export default config;
