/**
 * What is drawn *over* the canvas: the legend, the proof line, and the detail card.
 *
 * The legend is not optional furniture. The picture encodes four node kinds as four
 * shapes and the clients as hues, and an audience that has not read the spec has
 * about five seconds to learn that mapping — so it is on screen permanently,
 * at the same size as everything else, rather than behind a hover.
 */

import type React from "react";

import type { AskResponse } from "../api";
import type { RenderNode } from "../graph";
import { palette } from "../theme";

export function Legend({ hues }: { hues: Map<string, string> }) {
  return (
    <div className="legend">
      <div className="legend-row">
        <span className="legend-item">
          <svg width="20" height="20" viewBox="0 0 20 20">
            <path d="M2 6h6l1.4-2.4H17a1 1 0 0 1 1 1V16a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1Z" fill={palette.muted} />
          </svg>
          Ordner
        </span>
        <span className="legend-item">
          <svg width="20" height="20" viewBox="0 0 20 20">
            <rect x="5" y="2" width="10" height="16" rx="2" fill={palette.muted} />
          </svg>
          Dokument
        </span>
        <span className="legend-item">
          <svg width="20" height="20" viewBox="0 0 20 20">
            <circle cx="10" cy="10" r="7" fill={palette.muted} />
            <circle cx="10" cy="10" r="2.6" fill={palette.background} />
          </svg>
          Abschnitt <span className="dim">(Kern = zitiert)</span>
        </span>
        <span className="legend-item">
          <svg width="20" height="20" viewBox="0 0 20 20">
            <path d="M10 2 18 10 10 18 2 10Z" fill={palette.entity} />
          </svg>
          Entität
        </span>
      </div>
      <div className="legend-row">
        <span className="legend-item legend-crossing">
          <svg width="30" height="20" viewBox="0 0 30 20">
            <line x1="2" y1="10" x2="28" y2="10" stroke={palette.crossing} strokeWidth="4" />
          </svg>
          <strong>Kante über die Hierarchie hinweg</strong>
        </span>
        <span className="legend-item">
          <svg width="20" height="20" viewBox="0 0 20 20">
            <circle cx="10" cy="10" r="8" fill="none" stroke={palette.crossing} strokeWidth="2" opacity="0.6" />
            <path d="M10 4 16 10 10 16 4 10Z" fill={palette.crossing} />
          </svg>
          Brücke
        </span>
      </div>
      <div className="legend-row legend-branches">
        {[...hues].map(([branch, hue]) => (
          <span key={branch} className="legend-item">
            <span className="legend-swatch" style={{ background: hue }} />
            {branch}
          </span>
        ))}
      </div>
    </div>
  );
}

/** The claim, as three numbers, above the picture that justifies them — and room
 * for the one control that belongs with them rather than over the canvas. */
export function ProofLine({
  response,
  children,
}: {
  response: AskResponse;
  children?: React.ReactNode;
}) {
  const { legs, subgraph, parameters } = response;
  return (
    <div className="proof">
      <span className="proof-figure">
        <strong>{subgraph.crossing_edges}</strong> Kanten über die Hierarchie hinweg
      </span>
      <span className="proof-figure">
        <strong>{legs.traversal_only}</strong>/{legs.context} Abschnitte nur über den Graphen
      </span>
      <span className="proof-figure">
        <strong>{subgraph.branches.length}</strong>{" "}
        {subgraph.branches.length === 1 ? "Kunde" : "Kunden"} im Bild
      </span>
      <span className="proof-params">
        Hops {parameters.hops} · top_k {parameters.top_k} · Kontext {parameters.max_chunks} ·{" "}
        {parameters.legs} / {parameters.mode}
      </span>
      {children}
    </div>
  );
}

/** Everything one node knows, for the question an audience actually asks: and that
 * one — what is it, where does it sit, how did we get to it? */
export function NodeDetail({ node, onClose }: { node: RenderNode; onClose: () => void }) {
  return (
    <div className="detail">
      <button type="button" className="detail-close" onClick={onClose} aria-label="Schließen">
        ×
      </button>
      <div className="detail-kind">
        {{ folder: "Ordner", document: "Dokument", chunk: "Abschnitt", entity: "Entität" }[node.kind]}
        {node.bridge ? <span className="badge badge-bridge">Brücke</span> : null}
      </div>
      <div className="detail-label">{node.label}</div>

      {node.ancestry_lines.length > 0 ? (
        <div className="detail-ancestry">
          {node.ancestry_lines.map((line) => (
            <div key={line}>{line}</div>
          ))}
        </div>
      ) : null}

      {node.kind === "entity" ? (
        <dl className="detail-facts">
          <dt>Typ</dt>
          <dd>{node.type}</dd>
          <dt>Gültig in</dt>
          <dd>{node.scope ? node.scope : "global — gehört zu keinem Ordner"}</dd>
          <dt>Hop</dt>
          <dd>{node.hop}</dd>
          <dt>Erwähnungen im Korpus</dt>
          <dd>{node.mentions}</dd>
          {node.matched ? (
            <>
              <dt>Aus der Frage</dt>
              <dd>
                „{node.matched_as}" · {node.match_how?.join(", ")}
                {node.similarity !== null && node.similarity !== undefined
                  ? ` · ${node.similarity.toFixed(3)}`
                  : ""}
              </dd>
            </>
          ) : null}
          {node.branches.length > 0 ? (
            <>
              <dt>Erwähnt aus</dt>
              <dd>{node.branches.join(" · ")}</dd>
            </>
          ) : null}
        </dl>
      ) : null}

      {node.kind === "chunk" ? (
        <>
          <dl className="detail-facts">
            <dt>Gefunden von</dt>
            <dd>{node.legs?.join(", ")}</dd>
            <dt>Rang</dt>
            <dd>
              {Object.entries(node.ranks ?? {})
                .map(([leg, rank]) => `${leg} ${rank}`)
                .join(" · ")}
            </dd>
            <dt>Seite</dt>
            <dd>{node.page ?? "—"}</dd>
          </dl>
          <div className="detail-text">{node.text}</div>
        </>
      ) : null}

      {node.kind === "document" ? <div className="detail-path">{node.path}</div> : null}
    </div>
  );
}
