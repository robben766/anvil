/** Source returned in the SSE `sources` event (one per retrieved chunk). */
export interface Source {
  /** 1-based index used for [n] references in the answer. */
  n: number;
  chunk_id: string;
  document_id: string;
  /** Verbatim excerpt from the chunk. */
  quote: string;
  /** Breadcrumb path of headings, e.g. "# Title > ## Section". */
  header_path: string;
  start_offset: number;
  end_offset: number;
  score: number;
}

/** Citation referenced in the SSE `done` event (subset of Source without score). */
export interface Citation {
  n: number;
  chunk_id: string;
  document_id: string;
  quote: string;
  header_path: string;
  start_offset: number;
  end_offset: number;
}

/** Payload of the SSE `done` event. */
export interface QueryDone {
  text: string;
  citations: Citation[];
}
