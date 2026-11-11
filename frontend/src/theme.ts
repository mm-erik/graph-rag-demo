/**
 * The palette, chosen for a projector at the back of a room.
 *
 * Two rules it is built around, and both matter more than taste here:
 *
 * **Gold belongs to one idea.** Bridge entities and `crosses_branches` edges are
 * the only things drawn in it. The moment a second concept borrows the colour, the
 * audience stops being able to read "it reached across the hierarchy" off the
 * picture, and that reading is the whole demo.
 *
 * **A branch is a hue.** Every folder, document and chunk is filled with the hue of
 * the top-level folder it sits under, so two clients read as two coloured clusters
 * from a distance, before anyone has looked at a single label. Entities take the
 * hue of the branch that names them — or gold, when more than one does.
 *
 * Everything else is deliberately desaturated. On a badly calibrated projector the
 * contrast that survives is light-against-dark, not hue-against-hue.
 */

export const palette = {
  background: "#070b16",
  panel: "#0e1526",
  panelRaised: "#16203a",
  border: "#243352",
  text: "#eef3ff",
  muted: "#9fb0d0",
  dim: "#64769a",
  /** Reserved. Bridges and crossing edges, nothing else. */
  crossing: "#ffc83d",
  crossingSoft: "#8a6a16",
  entity: "#9db1d6",
} as const;

/** Structural edges: present, never competing. */
export const edgeColors = {
  contains: "#39486b",
  has_chunk: "#39486b",
  mentions: "#5d7bb5",
  relates: "#7a6ab5",
} as const;

/**
 * Branch hues, in the order branches are assigned them.
 *
 * Four, because the corpus has four clients and a fifth would mean the hierarchy
 * changed. Distinguishable from each other *and* from gold, which rules out the
 * oranges a four-colour palette would otherwise reach for.
 */
export const branchHues = [
  "#4fc3f7",
  "#6fcf97",
  "#bb86fc",
  "#4dd0e1",
] as const;

export const UNPLACED = "#8291b0";

/**
 * The width of the node/edge detail card, in CSS pixels.
 *
 * Here rather than only in `styles.css` because two things need it and they must
 * not drift: the card's own layout, and the camera, which shifts the centre of a
 * focused subgraph left by half this so that selecting a node does not park it
 * underneath the card describing it.
 */
export const DETAIL_WIDTH = 384;
