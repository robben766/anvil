import type { DebugItem, DebugFrame } from "@/lib/types";

interface Props {
  frame: DebugFrame;
}

/**
 * Three-column grid showing the raw retrieval internals:
 *   Dense (vector) | Sparse (BM25) | Fused (RRF)
 *
 * Row background:
 *   - chunk_id appears in BOTH dense and sparse → white
 *   - chunk_id only in dense                   → light blue
 *   - chunk_id only in sparse                  → light orange
 * Pure display — no side effects.
 */
export default function DebugPanel({ frame }: Props) {
  const { dense, sparse, fused } = frame;

  // Build sets for intersection/difference computation
  const denseIds = new Set(dense.map((d) => d.chunk_id));
  const sparseIds = new Set(sparse.map((s) => s.chunk_id));

  function rowBg(item: DebugItem, col: "dense" | "sparse"): string {
    const inDense = denseIds.has(item.chunk_id);
    const inSparse = sparseIds.has(item.chunk_id);
    if (inDense && inSparse) return "bg-white";
    if (col === "dense" && !inSparse) return "bg-blue-50";
    if (col === "sparse" && !inDense) return "bg-orange-50";
    return "bg-white";
  }

  return (
    <div className="grid grid-cols-3 gap-2 text-xs font-mono">
      {/* Dense column */}
      <div className="rounded border border-slate-200 overflow-hidden">
        <div className="bg-slate-100 px-2 py-1 font-semibold text-slate-700 text-center">
          向量召回
        </div>
        {dense.length === 0 ? (
          <div className="px-2 py-1 text-slate-400 text-center">—</div>
        ) : (
          dense.map((item) => (
            <div
              key={item.chunk_id}
              className={`flex items-baseline justify-between px-2 py-1 border-t border-slate-100 ${rowBg(item, "dense")}`}
            >
              <span className="text-slate-700 truncate max-w-[70%]">
                <span className="font-semibold text-slate-500">
                  #{item.rank}
                </span>{" "}
                {item.quote_head}
              </span>
              <span className="text-slate-400 shrink-0 tabular-nums ml-1">
                {item.score.toFixed(4)}
              </span>
            </div>
          ))
        )}
      </div>

      {/* Sparse column */}
      <div className="rounded border border-slate-200 overflow-hidden">
        <div className="bg-slate-100 px-2 py-1 font-semibold text-slate-700 text-center">
          BM25 召回
        </div>
        {sparse.length === 0 ? (
          <div className="px-2 py-1 text-slate-400 text-center">—</div>
        ) : (
          sparse.map((item) => (
            <div
              key={item.chunk_id}
              className={`flex items-baseline justify-between px-2 py-1 border-t border-slate-100 ${rowBg(item, "sparse")}`}
            >
              <span className="text-slate-700 truncate max-w-[70%]">
                <span className="font-semibold text-slate-500">
                  #{item.rank}
                </span>{" "}
                {item.quote_head}
              </span>
              <span className="text-slate-400 shrink-0 tabular-nums ml-1">
                {item.score.toFixed(4)}
              </span>
            </div>
          ))
        )}
      </div>

      {/* Fused (RRF) column */}
      <div className="rounded border border-slate-200 overflow-hidden">
        <div className="bg-slate-100 px-2 py-1 font-semibold text-slate-700 text-center">
          RRF 融合
        </div>
        {fused.length === 0 ? (
          <div className="px-2 py-1 text-slate-400 text-center">—</div>
        ) : (
          fused.map((item) => {
            const contrib = item.contributions;
            const parts: string[] = [];
            if (contrib) {
              if (contrib.dense !== null) parts.push(`d#${contrib.dense}`);
              if (contrib.sparse !== null) parts.push(`s#${contrib.sparse}`);
            }
            const contribLabel = parts.join(" + ");

            return (
              <div
                key={item.chunk_id}
                className="px-2 py-1 border-t border-slate-100 bg-white"
              >
                <div className="flex items-baseline justify-between">
                  <span className="text-slate-700 truncate max-w-[70%]">
                    <span className="font-semibold text-slate-500">
                      #{item.rank}
                    </span>{" "}
                    {item.quote_head}
                  </span>
                  <span className="text-slate-400 shrink-0 tabular-nums ml-1">
                    {item.score.toFixed(4)}
                  </span>
                </div>
                {contribLabel && (
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    {contribLabel}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
