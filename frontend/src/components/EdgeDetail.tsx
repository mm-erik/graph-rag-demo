/**
 * The card that justifies the colour.
 *
 * A gold edge is an assertion — *this connection leaves the hierarchy* — and an
 * assertion the audience cannot interrogate is one they have to take on trust. So
 * hovering one says which two branches it spans, what relation it is, and the
 * sentence from the corpus the extraction took it from.
 */

import type { RenderEdge } from "../graph";

const KINDS: Record<string, string> = {
  contains: "enthält",
  has_chunk: "hat Abschnitt",
  mentions: "erwähnt",
  relates: "Relation",
};

export function EdgeDetail({ edge }: { edge: RenderEdge }) {
  const from = edge.source;
  const to = edge.target;
  // An endpoint is still a plain id until the simulation resolves it to its node;
  // a card can only be hovered after that, so in practice this reads the label.
  const name = (end: typeof from) => (typeof end === "string" ? end : end.label);
  return (
    <div className={`detail detail-edge${edge.crosses_branches ? " detail-crossing" : ""}`}>
      <div className="detail-kind">
        {KINDS[edge.kind] ?? edge.kind}
        {edge.crosses_branches ? (
          <span className="badge badge-bridge">über die Hierarchie hinweg</span>
        ) : null}
      </div>
      <div className="detail-label">
        {name(from)} <span className="dim">—{edge.label ? ` ${edge.label} ` : " "}→</span>{" "}
        {name(to)}
      </div>
      {edge.branches.length > 0 ? (
        <div className="detail-ancestry">
          Spannt: {edge.branches.join(" · ")}
        </div>
      ) : null}
      {edge.evidence ? <div className="detail-text">„{edge.evidence}"</div> : null}
    </div>
  );
}
