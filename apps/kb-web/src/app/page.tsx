"use client";

import { useState } from "react";
import DocumentPanel from "@/components/DocumentPanel";
import ChatPanel from "@/components/ChatPanel";
import CitationPanel from "@/components/CitationPanel";
import type { Citation } from "@/lib/types";

export default function Home() {
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);

  return (
    <div className="flex h-full flex-col">
      {/* Top bar */}
      <header className="border-b border-slate-200 bg-white px-6 py-3">
        <h1 className="text-lg font-semibold text-slate-800">
          anvil-kb 知识库
        </h1>
      </header>

      {/* Main content: left 1/3 panel + right 2/3 chat */}
      <main className="relative flex flex-1 overflow-hidden">
        {/* Left: document panel */}
        <aside className="w-1/3 overflow-y-auto border-r border-slate-200 p-4">
          <DocumentPanel />
        </aside>

        {/* Right: chat panel */}
        <section className="flex w-2/3 flex-col overflow-hidden">
          <ChatPanel onCite={(c) => setActiveCitation(c)} />
        </section>

        {/* Citation overlay panel (fixed, above content) */}
        <CitationPanel
          citation={activeCitation}
          onClose={() => setActiveCitation(null)}
        />
      </main>
    </div>
  );
}
