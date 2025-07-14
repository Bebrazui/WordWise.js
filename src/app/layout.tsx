import type {Metadata} from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'WordWise.js',
  description: 'A custom ML framework for training text-based AI in the browser.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
