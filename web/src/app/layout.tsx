import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Arkain Atlas — CRE Deal Intelligence",
  description: "AI-powered commercial real estate deal intelligence platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
