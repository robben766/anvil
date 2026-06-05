import DocumentPanel from "@/components/DocumentPanel";

export default function Home() {
  return (
    <div className="flex h-full flex-col">
      {/* Top bar */}
      <header className="border-b border-slate-200 bg-white px-6 py-3">
        <h1 className="text-lg font-semibold text-slate-800">
          anvil-kb 知识库
        </h1>
      </header>

      {/* Main content: left 1/3 panel + right 2/3 placeholder */}
      <main className="flex flex-1 overflow-hidden">
        {/* Left: document panel */}
        <aside className="w-1/3 overflow-y-auto border-r border-slate-200 p-4">
          <DocumentPanel />
        </aside>

        {/* Right: QA placeholder */}
        <section className="flex w-2/3 items-center justify-center p-6 text-slate-400">
          <div className="rounded border border-dashed border-slate-300 px-8 py-10 text-center">
            <p className="text-sm">问答区（KB-M1b Task B4）</p>
          </div>
        </section>
      </main>
    </div>
  );
}
