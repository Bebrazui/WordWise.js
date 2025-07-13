import type {NextConfig} from 'next';

const nextConfig: NextConfig = {
  /* config options here */
  typescript: {
    ignoreBuildErrors: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'placehold.co',
        port: '',
        pathname: '/**',
      },
    ],
  },
  webpack: (config, { isServer }) => {
    // Exclude specific modules from client-side bundle
    if (!isServer) {
        config.externals = [
            ...config.externals,
            'sharp',
            'onnxruntime-node',
        ];
    }
    return config;
  },
};

export default nextConfig;
