/**
 * Drives the built UI against the running service and writes screenshots.
 *
 * This exists for one acceptance criterion that cannot be checked any other way:
 * *legible on a projector at a distance, verified at presentation scale, not just
 * on a laptop.* A judgement about legibility has to be made by looking, and looking
 * at 1280×720 with the stage control turned up is a different picture from looking
 * at a developer's monitor. It is a verification tool, not a test — nothing here
 * asserts, and the project's testing posture (see the README) is deliberately
 * narrow.
 *
 *   just serve                       # in another terminal, with the corpus ingested
 *   node verify.mjs 1280 720 stage   # a projector, at presentation scale
 *   node verify.mjs 1600 1000 ask    # a laptop, default view
 *
 * Actions: empty | ask | crossings | cite | stage | followup.
 * Screenshots land in ./shots, which is not committed.
 */

import { mkdir } from "node:fs/promises";
import puppeteer from "puppeteer-core";

const CHROME =
  process.env.CHROME ??
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const ORIGIN = process.env.ORIGIN ?? "http://127.0.0.1:8000/";
const [, , width = "1600", height = "1000", action = "ask"] = process.argv;
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

await mkdir("shots", { recursive: true });

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: "new",
  args: [`--window-size=${width},${height}`],
  defaultViewport: { width: Number(width), height: Number(height) },
});
const page = await browser.newPage();
const errors = [];
page.on("console", (message) => {
  if (message.type() === "error") errors.push(message.text());
});
page.on("pageerror", (error) => errors.push(String(error)));
await page.goto(ORIGIN, { waitUntil: "networkidle0" });

const clickText = (selector, text) =>
  page.evaluate(
    (query, label) => {
      const element = [...document.querySelectorAll(query)].find(
        (candidate) => candidate.textContent.trim() === label,
      );
      if (!element) throw new Error(`no ${query} labelled ${label}`);
      element.click();
    },
    selector,
    text,
  );

if (action !== "empty") {
  await clickText(".suggestions button", "Zielfrage");
  await page.waitForSelector(".answer", { timeout: 180_000 });
  // The force layout settles and the camera frames it over the following seconds.
  await wait(8000);
}
if (action === "crossings") {
  await clickText(".crossings", "Nur die Übergänge");
  await wait(1200);
}
if (action === "cite") {
  await page.evaluate(() => document.querySelectorAll(".cite")[1].click());
  await wait(1200);
}
if (action === "stage") {
  await clickText(".segmented button", "groß");
  // Longer than it looks necessary: raising the scale re-heats the collide force,
  // and the camera re-frames on a timer once the nodes have stopped travelling.
  await wait(9000);
}
if (action === "followup") {
  await page.type(".composer input", "Und wer hat sonst noch daran gearbeitet?");
  await page.click(".composer button");
  await page.waitForFunction(
    () => document.querySelectorAll(".turn-chip").length === 2,
    { timeout: 180_000 },
  );
  await wait(8000);
}

const seen = await page.evaluate(() => ({
  proof: document.querySelector(".proof")?.innerText,
  rewrite: document.querySelector(".rewrite")?.innerText,
  sources: document.querySelectorAll(".source").length,
  citations: document.querySelectorAll(".cite").length,
}));
const path = `shots/${action}-${width}x${height}.png`;
await page.screenshot({ path });
console.log(JSON.stringify({ path, seen, errors }, null, 1));
await browser.close();
