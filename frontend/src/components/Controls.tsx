/**
 * The knobs, on screen, where they can be turned mid-sentence.
 *
 * Hop depth is the one this exists for. It is the parameter that decides whether
 * the picture is a readable argument or a hairball, it is different for different
 * questions, and the spec says plainly it will be changed live on stage. The rest
 * are here because an operator who can see only one of five knobs cannot explain
 * which one produced the picture.
 *
 * `Dichte` and `Bühne` are the two that change nothing on the server: the first
 * drops entity relations that neither cross a branch nor touch a bridge, the
 * second scales the whole drawing for a room rather than a desk.
 */

import type { Legs, Mode, Tuning } from "../api";

interface Props {
  tuning: Tuning;
  onTuning: (tuning: Tuning) => void;
  window: number;
  onWindow: (window: number) => void;
  dense: boolean;
  onDense: (dense: boolean) => void;
  hidden: { edges: number; nodes: number };
  scale: number;
  onScale: (scale: number) => void;
  busy: boolean;
}

function Slider({
  label,
  hint,
  value,
  min,
  max,
  onChange,
  emphasis,
}: {
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
  emphasis?: boolean;
}) {
  return (
    <label className={`knob${emphasis ? " knob-emphasis" : ""}`} title={hint}>
      <span className="knob-label">
        {label}
        <strong>{value}</strong>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={1}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

/** One row of mutually exclusive buttons. Generic over the value because the two
 * display-only knobs — density and stage scale — choose a boolean and a number
 * while the rest choose one of the service's enums. */
function Choice<T extends string | number | boolean>({
  label,
  hint,
  value,
  options,
  note,
  onChange,
}: {
  label: string;
  hint: string;
  value: T;
  options: [T, string][];
  note?: string;
  onChange: (value: T) => void;
}) {
  return (
    <div className="knob" title={hint}>
      <span className="knob-label">{label}</span>
      <div className="segmented">
        {options.map(([option, text]) => (
          <button
            key={String(option)}
            type="button"
            className={option === value ? "on" : ""}
            onClick={() => onChange(option)}
          >
            {text}
          </button>
        ))}
      </div>
      {note ? <span className="knob-note">{note}</span> : null}
    </div>
  );
}

export function Controls({
  tuning,
  onTuning,
  window,
  onWindow,
  dense,
  onDense,
  hidden,
  scale,
  onScale,
  busy,
}: Props) {
  const set = (patch: Partial<Tuning>) => onTuning({ ...tuning, ...patch });
  return (
    <div className={`controls${busy ? " controls-busy" : ""}`}>
      <Slider
        label="Hops"
        hint="Wie weit der Entitätengraph abgelaufen wird. Der Regler, der live gedreht wird: 1 ergibt ein Bild, das aus der letzten Reihe lesbar ist, 2 zeigt mehr Verbindungen."
        value={tuning.hops}
        min={0}
        max={5}
        onChange={(hops) => set({ hops })}
        emphasis
      />
      <Slider
        label="Treffer je Bein"
        hint="top_k: wie viele Abschnitte jedes Retrieval-Bein zurückgibt."
        value={tuning.top_k}
        min={1}
        max={30}
        onChange={(top_k) => set({ top_k })}
      />
      <Slider
        label="Kontext"
        hint="max_chunks: wie viele Abschnitte die Antwort höchstens lesen darf — und damit, wie viele im Bild stehen."
        value={tuning.max_chunks}
        min={1}
        max={30}
        onChange={(max_chunks) => set({ max_chunks })}
      />
      <Choice<Legs>
        label="Beine"
        hint="Beide Beine, nur die lexikalisch/semantische Suche, oder nur die Traversierung. Dieselbe Frage einmal ohne Graph zu stellen ist die Demo."
        value={tuning.legs}
        options={[
          ["both", "beide"],
          ["search", "Suche"],
          ["traversal", "Graph"],
        ]}
        onChange={(legs) => set({ legs })}
      />
      <Choice<Mode>
        label="Suche"
        hint="Hybrid aus Vektor und Volltext, oder eines davon allein."
        value={tuning.mode}
        options={[
          ["hybrid", "hybrid"],
          ["vector", "Vektor"],
          ["fulltext", "Volltext"],
        ]}
        onChange={(mode) => set({ mode })}
      />
      <Slider
        label="Gedächtnis"
        hint="Wie viele vorige Züge eine Rückfrage auflösen. 0 schaltet das Gedächtnis ab — so scheitert dieselbe Rückfrage sichtbar."
        value={window}
        min={0}
        max={10}
        onChange={onWindow}
      />
      <Choice<boolean>
        label="Dichte"
        hint="Lesbar zeigt die Relationen, die eine Grenze überschreiten, und die zwischen Entitäten, die ein gezeigter Abschnitt auch nennt. Der Rest — der Fächer hinaus zu Entitäten, die hier niemand nennt — wird ausgeblendet."
        value={dense}
        options={[
          [false, "lesbar"],
          [true, "alles"],
        ]}
        note={
          !dense && hidden.edges > 0
            ? `${hidden.edges} Relationen, ${hidden.nodes} Entitäten verborgen`
            : undefined
        }
        onChange={onDense}
      />
      <Choice<number>
        label="Bühne"
        hint="Maßstab für den Beamer: Knoten, Beschriftungen und die Antwort werden größer gezeichnet — die Bedienleiste nicht, sonst frisst sie die Fläche, die das Bild gewinnen sollte."
        value={scale}
        options={[
          [1, "Laptop"],
          [1.3, "Saal"],
          [1.6, "groß"],
        ]}
        onChange={onScale}
      />
    </div>
  );
}
