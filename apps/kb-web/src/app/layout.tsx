import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "anvil-kb 知识库",
  description: "anvil KB document management",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN" className="h-full">
      <body className="h-full bg-white text-slate-900 antialiased">
        {children}
      </body>
    </html>
  );
}
