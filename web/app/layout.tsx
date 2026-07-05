import type { Metadata } from "next";

import { AppSidebar } from "@/components/layout/AppSidebar";

import "./globals.css";

export const metadata: Metadata = {
  title: "The A-Files — audio steganography research platform",
  description:
    "Experiment dashboard for audio steganography research: benchmarks, robustness, perceptual quality, capacity and method comparison.",
};

// Applies the persisted (or system) theme before first paint to avoid a flash.
const themeInitScript = `
(function () {
  try {
    var stored = localStorage.getItem("taf-theme");
    var dark = stored ? stored === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.classList.toggle("dark", dark);
  } catch (e) {}
})();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
      </head>
      <body className="min-h-screen font-sans antialiased">
        <div className="flex min-h-screen">
          <AppSidebar />
          <main className="flex-1 overflow-y-auto p-8">{children}</main>
        </div>
      </body>
    </html>
  );
}
