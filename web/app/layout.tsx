import type { Metadata } from "next";

import { MotionProvider } from "@/components/motion-provider";
import { AppShell } from "@/components/shell";
import { I18nProvider } from "@/lib/i18n";

import "./globals.css";

export const metadata: Metadata = {
  title: "The A-Files — research platform",
  description:
    "Design, run and analyse experiments on audio steganography and watermarking: imperceptibility, robustness, capacity and security.",
};

// Applies the persisted (or system) theme and language before first paint.
const initScript = `
(function () {
  try {
    var stored = localStorage.getItem("taf-theme");
    var dark = stored ? stored === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.classList.toggle("dark", dark);
    var locale = localStorage.getItem("taf-locale");
    if (locale === "pl") document.documentElement.lang = "pl";
  } catch (e) {}
})();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: initScript }} />
      </head>
      <body className="min-h-screen font-sans antialiased">
        <I18nProvider>
          <MotionProvider>
            <AppShell>{children}</AppShell>
          </MotionProvider>
        </I18nProvider>
      </body>
    </html>
  );
}
