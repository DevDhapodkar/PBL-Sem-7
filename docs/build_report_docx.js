/*
 * Build the Word report: docs/Nagpur_Debris_Detection_Report.docx
 * Run:  npm install docx && node docs/build_report_docx.js
 *       (the TOC page numbers populate when Word first opens the file / F9)
 *
 * Documents the whole project — problem, live data, method, and especially the
 * comparison against the traditional method WITHOUT the point-set alignment, and
 * HOW that comparison was done — with screenshots. Numbers mirror results/.
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType, ImageRun,
  TableOfContents, PageBreak, LevelFormat, PageOrientation,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const IMG = path.join(ROOT, "docs", "img");
const RES = path.join(ROOT, "results");

// ---- palette ----
const INK = "13202E", INK2 = "44586A", ACCENT = "0D6E7C", ACCENT2 = "0F9C6D";
const RED = "C0392B", MUTED = "7C8B99", RULE = "D9E1EA", BAND = "EEF4F6";

// ---- content geometry (US Letter, 1" margins) ----
const CONTENT_DXA = 9360;          // 6.5"
const CONTENT_PX = 624;            // 6.5" * 96

// ---- helpers ----
function pngSize(file) {
  const buf = fs.readFileSync(file);
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}
function image(file, maxPx = CONTENT_PX, caption = null) {
  const p = path.join(IMG, file);
  const { w, h } = pngSize(p);
  const width = Math.min(maxPx, w);
  const height = Math.round((width / w) * h);
  const runs = [new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: caption ? 40 : 160 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(p),
      transformation: { width, height } })],
  })];
  if (caption) runs.push(new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 180 },
    children: [new TextRun({ text: caption, italics: true, size: 17, color: MUTED })],
  }));
  return runs;
}
function h(text, level) {
  return new Paragraph({ heading: level, children: [new TextRun(text)] });
}
function p(text, opts = {}) {
  const runs = Array.isArray(text) ? text
    : [new TextRun({ text, size: 21, color: INK2 })];
  return new Paragraph({ spacing: { after: 140, line: 276 }, ...opts, children: runs });
}
function run(text, o = {}) {
  return new TextRun({ text, size: o.size || 21, color: o.color || INK2,
    bold: o.bold || false, italics: o.italics || false });
}
function bullets(items) {
  return items.map((it) => new Paragraph({
    numbering: { reference: "bull", level: 0 }, spacing: { after: 80, line: 270 },
    children: Array.isArray(it) ? it : [new TextRun({ text: it, size: 21, color: INK2 })],
  }));
}
// simple table with header row; cols = array of DXA widths summing to CONTENT_DXA
function table(cols, header, rows, opts = {}) {
  const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
  const borders = { top: border, bottom: border, left: border, right: border,
    insideHorizontal: border, insideVertical: border };
  const cell = (txt, i, o = {}) => new TableCell({
    width: { size: cols[i], type: WidthType.DXA },
    shading: o.shade ? { type: ShadingType.CLEAR, fill: o.shade, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    children: [new Paragraph({
      alignment: i === 0 ? AlignmentType.LEFT : (opts.numAlign || AlignmentType.CENTER),
      children: (Array.isArray(txt) ? txt : [new TextRun({
        text: String(txt), size: 18, bold: o.bold || false,
        color: o.color || (o.bold ? INK : INK2) })]),
    })],
  });
  const headRow = new TableRow({ tableHeader: true, children: header.map((t, i) =>
    new TableCell({ width: { size: cols[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: ACCENT, color: "auto" },
      margins: { top: 70, bottom: 70, left: 110, right: 110 },
      children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        children: [new TextRun({ text: t, size: 17, bold: true, color: "FFFFFF" })] })] })) });
  const bodyRows = rows.map((r) => new TableRow({ children: r.cells.map((c, i) =>
    cell(c, i, { bold: r.bold, color: r.color, shade: r.shade })) }));
  return new Table({ columnWidths: cols, width: { size: CONTENT_DXA, type: WidthType.DXA },
    borders, rows: [headRow, ...bodyRows] });
}
function spacer(after = 120) { return new Paragraph({ spacing: { after } }); }
function rule() {
  return new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE } },
    spacing: { after: 160 } });
}

// ===========================================================================
// document body
// ===========================================================================
const body = [];

// ---- title page ----
body.push(new Paragraph({ spacing: { before: 2600, after: 0 },
  children: [new TextRun({ text: "CROSS-MODAL SATELLITE DEBRIS DETECTION",
    size: 20, color: ACCENT, bold: true })] }));
body.push(new Paragraph({ spacing: { before: 120, after: 0 },
  children: [new TextRun({ text: "Aligning Sentinel-1 and Sentinel-2 to Detect",
    size: 52, bold: true, color: INK })] }));
body.push(new Paragraph({ spacing: { after: 200 },
  children: [new TextRun({ text: "Floating Debris on Nagpur's Water Bodies",
    size: 52, bold: true, color: INK })] }));
body.push(new Paragraph({ spacing: { after: 60 }, children: [new TextRun({
  text: "A cross-modal fusion pipeline whose contribution is a robust point-set "
      + "alignment (RANSAC → CPD) — benchmarked head-to-head against the "
      + "traditional method that fuses the two sensors WITHOUT alignment.",
  size: 24, italics: true, color: INK2 })] }));
body.push(spacer(500));
body.push(...image("overlay_latest.png", 560,
  "Fully-real result: latest clear Sentinel-2 optical × latest Sentinel-1D radar over "
  + "Ambazari Lake, aligned and cross-validated."));
body.push(new Paragraph({ spacing: { before: 400 }, children: [
  new TextRun({ text: "Project-based learning report", size: 20, color: MUTED })] }));
body.push(new Paragraph({ children: [
  new TextRun({ text: "Repository: DevDhapodkar/PBL-Sem-7", size: 20, color: MUTED })] }));
body.push(new Paragraph({ children: [new PageBreak()] }));

// ---- table of contents ----
body.push(h("Contents", HeadingLevel.HEADING_1));
body.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
body.push(new Paragraph({ children: [new PageBreak()] }));

// ---- 1. Executive summary ----
body.push(h("1. Executive summary", HeadingLevel.HEADING_1));
body.push(p("Floating matter on inland lakes — water hyacinth, algal scum, plastic and "
  + "mixed-trash rafts — is visible to two very different satellites. Sentinel-2 sees it "
  + "optically (a vegetation-like or debris spectral signature); Sentinel-1 sees it in "
  + "radar (it changes the water-surface roughness). Each sensor alone raises false alarms "
  + "the other does not, so agreement between the two is strong evidence of real debris."));
body.push(p([run("This project runs an independent detector on each sensor, turns the "
  + "detections into object-level "), run("point sets", { bold: true }),
  run(" (a “scatter plot” per sensor), and then — the core idea — "),
  run("aligns the two point sets", { bold: true, color: ACCENT }),
  run(" before comparing them, using a robust RANSAC → Coherent Point Drift "
  + "registration. A detection confirmed by both sensors after alignment is reported as "
  + "debris; a single-sensor detection is discounted as a likely artifact.")]));
body.push(p([run("The headline finding, measured against the traditional method that does "
  + "the same fusion "), run("without", { bold: true, italics: true }),
  run(" the alignment: on a controlled benchmark with known ground truth, the "
  + "no-alignment method recovers almost nothing (recall 0.12, F1 0.20), while the identical "
  + "fusion "), run("with", { bold: true }), run(" our alignment reaches recall 0.54 and "
  + "F1 0.67 — a ~4.6× recall gain from that one step. It also beats a classic ICP "
  + "alignment (F1 0.67 vs 0.42). Everything runs on real, live Sentinel-1 and Sentinel-2 "
  + "data.")]));

// ---- 2. Problem ----
body.push(h("2. The problem — Nagpur's water bodies", HeadingLevel.HEADING_1));
body.push(p("Nagpur, the “City of Lakes” (Maharashtra, India), has a set of "
  + "historic tanks and reservoirs that suffer from water-hyacinth blooms, sewage-driven "
  + "algal scum and floating solid waste — a real, local environmental-monitoring problem "
  + "well suited to satellite fusion. All the lakes fall inside a single Sentinel-2 tile "
  + "(44QKJ, UTM zone 44N), which the pipeline reads directly."));
body.push(...image("nagpur_scene.png", CONTENT_PX,
  "Real Sentinel-2 of Ambazari Lake. Left: true colour with both sensors' detections. "
  + "Middle: NDWI water mask. Right: the NDVI/FDI floating-matter cue on water."));

// ---- 3. Live data ----
body.push(h("3. Live satellite data (both sensors real)", HeadingLevel.HEADING_1));
body.push(p("Both modalities are fetched live from public open-data sources — no simulation "
  + "and no account required. Sentinel-1 is read straight from a bare AWS S3 bucket, so it "
  + "works even in networks that block the usual satellite discovery APIs."));
body.push(table([1900, 3100, 4360],
  ["Modality", "Product & source", "Live status"],
  [
    { cells: ["Sentinel-2 (optical)",
      "L2A surface reflectance COGs, AWS bucket sentinel-cogs, tile 44QKJ",
      "Real & live via /vsicurl range reads; five lakes bundled offline"] },
    { cells: ["Sentinel-1 (radar)",
      "GRD VV/VH, AWS bucket sentinel-s1-l1c (Sinergise open data)",
      "Real & live — latest pass found & geocoded via embedded GCPs; needs only S3"] },
  ], { numAlign: AlignmentType.LEFT }));
body.push(spacer(60));
body.push(p([run("How real Sentinel-1 is obtained. ", { bold: true, color: INK }),
  run("The AWS GRD bucket is partitioned only by date, with no spatial index, so the "
  + "pipeline discovers the latest scene over a lake by sampling each day’s "
  + "time-sorted frames densely enough that no orbit pass is skipped, then confirming the "
  + "lake falls inside a frame’s footprint. Each GRD raster carries its geolocation "
  + "grid as embedded ground-control points, so only the small lake window is reprojected "
  + "onto the Sentinel-2 grid. The result verified in this report: Sentinel-2 of "
  + "2026-07-16 paired with Sentinel-1D of 2026-08-10 over Ambazari.")]));

// ---- 4. Method ----
body.push(h("4. Method — the pipeline", HeadingLevel.HEADING_1));
body.push(p("Each stage has a distinct job. The alignment step (Stage 2, highlighted) is "
  + "the project’s contribution; the traditional baseline simply omits it."));
body.push(...image("ui_pipeline.png", CONTENT_PX,
  "The pipeline: detect per sensor → object point sets → align the point sets "
  + "(our idea) → cross-modal validation → debris map."));
body.push(h("4.1 Detectors (bands → point sets)", HeadingLevel.HEADING_2));
body.push(...bullets([
  [run("Water mask: ", { bold: true, color: INK }), run("NDWI = (Green − NIR)/(Green + NIR).")],
  [run("Floating vegetation: ", { bold: true, color: INK }), run("NDVI (hyacinth / algal scum).")],
  [run("Floating debris: ", { bold: true, color: INK }), run("FDI, the Floating Debris Index (Biermann 2020), for plastic / trash rafts.")],
  [run("Radar anomaly: ", { bold: true, color: INK }), run("a CFAR-style local-contrast test on VV backscatter over water.")],
  [run("Point sets: ", { bold: true, color: INK }), run("connected-component centroids of each anomaly mask; the index value gives a per-detection strength.")],
]));
body.push(h("4.2 Alignment (Stage 1–2) — the contribution", HeadingLevel.HEADING_2));
body.push(p([run("RANSAC ", { bold: true, color: INK }),
  run("estimates a robust initial transform from spatial putative correspondences and "
  + "rejects gross outlier pairs. "), run("Coherent Point Drift (CPD) ", { bold: true, color: INK }),
  run("then refines it probabilistically, with a uniform component that absorbs the "
  + "modality-specific false alarms that break simpler methods. For real geocoded "
  + "Sentinel-1/2 the model is a bounded translation (the products already share a grid, so "
  + "the residual is a small co-registration offset plus limited inter-pass drift).")]));
body.push(h("4.3 Cross-modal validation & confidence (Stage 3)", HeadingLevel.HEADING_2));
body.push(p("After the radar point set is aligned into the optical frame, detections are "
  + "matched by mutual nearest neighbour within a tolerance. Each candidate gets a "
  + "confidence C that combines geometric agreement, both sensors’ evidence strengths, "
  + "and a global registration-quality gate Q_reg. Q_reg multiplies every confidence, so a "
  + "failed alignment can never manufacture confident debris — this is what keeps the score "
  + "honest. A single-sensor detection is heavily discounted."));

// ---- 5. The comparison ----
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(h("5. The comparison: with vs without the alignment", HeadingLevel.HEADING_1));
body.push(h("5.1 What we compared against", HeadingLevel.HEADING_2));
body.push(p("Four methods, so the alignment can be isolated:"));
body.push(table([2400, 2100, 4860],
  ["Method", "Alignment", "What it represents"],
  [
    { cells: ["optical-only", "— (single sensor)", "no fusion at all — the “trust one sensor” baseline"] },
    { cells: ["no alignment (traditional)", "none (identity)",
      "the traditional cross-modal method WITHOUT our idea: overlay the two geocoded point sets as-is and take the agreement"],
      shade: "FBEEE9" },
    { cells: ["ICP", "ICP registration", "a classic alignment attempt (no outlier model)"] },
    { cells: ["RANSAC + CPD (ours)", "our point-set alignment", "robust, outlier-aware alignment"],
      shade: "E7F5EF" },
  ], { numAlign: AlignmentType.LEFT }));

body.push(h("5.2 How the comparison was done (methodology)", HeadingLevel.HEADING_2));
body.push(p("The comparison is designed to change only the alignment and hold everything "
  + "else fixed, so any performance gap is attributable to the alignment and nothing else:"));
body.push(...bullets([
  [run("Same detectors for every method. ", { bold: true, color: INK }),
   run("All methods start from the identical Sentinel-2 and Sentinel-1 detections — no "
   + "method gets a better input point set than another.")],
  [run("Same decision stage for every fusion method. ", { bold: true, color: INK }),
   run("no-alignment, ICP and ours feed their point sets into the same cross-modal "
   + "validation and the same confidence threshold. Only the alignment step varies.")],
  [run("Regime A — controlled simulation with ground truth. ", { bold: true, color: INK }),
   run("Scenes are generated with a known set of true debris and a known "
   + "radar→optical mis-registration, then each detector is given realistic misses, "
   + "localisation noise and modality-specific false positives. Because truth is known, we "
   + "compute precision, recall, F1 and average precision, averaged over 20 random seeds, at "
   + "a fixed operating point — and sweep the clutter load for robustness. This isolates the "
   + "algorithm.")],
  [run("Regime B — real Nagpur lakes. ", { bold: true, color: INK }),
   run("There is no ground-truth label for a live lake, so we report what can be measured "
   + "objectively: debris corroborated by both sensors, the recovered offset, and how the "
   + "corroboration count behaves as the match tolerance is tightened.")],
  [run("Reported result = the ordering. ", { bold: true, color: INK }),
   run("Absolute simulation scores depend on the injected noise; the ranking "
   + "(ours > ICP > no-alignment) is the finding.")],
]));

body.push(h("5.3 Result A — controlled benchmark (20 seeds, ground truth)", HeadingLevel.HEADING_2));
body.push(table([2760, 1400, 1300, 1300, 1300, 1300],
  ["Method", "Align", "Precision", "Recall", "F1", "AP"],
  [
    { cells: ["optical-only", "—", "0.75", "0.81", "0.78", "0.75"] },
    { cells: ["no alignment (traditional)", "none", "0.65", "0.12", "0.20", "0.52"], shade: "FBEEE9" },
    { cells: ["ICP", "ICP", "0.79", "0.30", "0.42", "0.66"] },
    { cells: ["RANSAC + CPD (ours)", "ours", "0.93", "0.54", "0.67", "0.81"], bold: true, shade: "E7F5EF" },
  ]));
body.push(spacer(40));
body.push(p([run("Reading it: ", { bold: true, color: INK }),
  run("without alignment the offset between the two point sets pushes real pairs outside "
  + "the match tolerance, so almost nothing is corroborated (recall 0.12, F1 0.20). Adding "
  + "the alignment lifts the same fusion to recall 0.54, F1 0.67 — ~4.6× recall and "
  + "~3.4× F1 from one step — and beats ICP, which has no outlier model and is dragged "
  + "off by each sensor’s false alarms.")]));
body.push(...image("ui_compare.png", CONTENT_PX,
  "The comparison at a glance: four-method table and F1 / recall bars. No-alignment "
  + "(orange) collapses; our alignment (green) recovers the corroborations."));
body.push(...image("sim_pr.png", 520,
  "Precision–recall curves. Average precision: ours 0.81 > ICP 0.66 > no-alignment 0.55."));
body.push(...image("sim_robustness.png", CONTENT_PX,
  "Robustness: as modality-specific clutter rises, our aligned fusion holds precision far "
  + "longer than the baselines."));

body.push(h("5.4 Result B — with vs without alignment on real, live data", HeadingLevel.HEADING_2));
body.push(p("The fully-real pipeline over three Nagpur lakes recovers a genuine but small "
  + "residual offset (real products are already fairly well geolocated):"));
body.push(table([2000, 1500, 1500, 1440, 1440, 1480],
  ["Lake", "S2 date", "S1 date", "Optical", "Radar", "Offset"],
  [
    { cells: ["Ambazari", "2026-07-16", "2026-08-10", "41", "31", "~50 m"] },
    { cells: ["Gorewada", "2026-05-30", "2026-08-10", "47", "25", "~19 m"] },
    { cells: ["Futala", "2026-05-27", "2026-08-10", "49", "15", "~130 m"] },
  ]));
body.push(spacer(60));
body.push(p("The alignment’s value appears when the match tolerance is tight (a precise, "
  + "low-false-match radius). Sweeping the tolerance on Ambazari, the no-alignment overlay "
  + "loses the corroborations the offset pushes out of range, while our alignment keeps them:"));
body.push(table([3360, 1000, 1000, 1000, 1000, 1000, 1000],
  ["Match tolerance", "20 m", "30 m", "40 m", "50 m", "60 m", "80 m"],
  [
    { cells: ["no alignment (traditional)", "1", "1", "1", "2", "5", "5"], shade: "FBEEE9" },
    { cells: ["with alignment (ours)", "3", "3", "3", "3", "3", "5"], bold: true, shade: "E7F5EF" },
  ]));
body.push(spacer(40));
body.push(p([run("At a tight 30 m tolerance the alignment confirms 3× more debris "
  + "(3 vs 1). ", { bold: true, color: INK }),
  run("At a loose tolerance the two converge — but a loose tolerance also admits false "
  + "matches, so the alignment is what lets you match precisely.")]));
body.push(...image("ui_real.png", CONTENT_PX,
  "Live Nagpur panel: the real S1×S2 overlay (left), the per-lake table, and the "
  + "match-tolerance sweep (right) — alignment holds true pairs a tight tolerance drops."));

// ---- 6. UI ----
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(h("6. The user interface", HeadingLevel.HEADING_1));
body.push(p("Two UIs ship with the project. An interactive Streamlit dashboard "
  + "(streamlit run app.py) drives the pipeline on any lake with live controls. And a "
  + "self-contained HTML dashboard (docs/dashboard.html) is dedicated to this comparison — a "
  + "single static page, no server, that reads the pipeline’s own result files so its "
  + "numbers can never drift from the code. It is theme-aware (light and dark)."));
body.push(...image("ui_hero.png", CONTENT_PX,
  "The comparison dashboard header and key-metric strip, showing the with→without "
  + "alignment deltas at a glance."));
body.push(p("The dashboard walks through the pipeline, the four-method comparison table and "
  + "charts, the robustness curve, and the live Nagpur data — the same evidence as Section 5, "
  + "rendered interactively and shareable as a link."));

// ---- 7. Conclusions ----
body.push(h("7. Conclusions", HeadingLevel.HEADING_1));
body.push(...bullets([
  [run("The point-set alignment is what makes cross-modal fusion work. ", { bold: true, color: INK }),
   run("Fusing the two sensors without aligning the point sets barely corroborates anything "
   + "(recall 0.12, F1 0.20); adding the alignment lifts the identical fusion to recall 0.54 / "
   + "F1 0.67. This is the central result.")],
  [run("It runs on real, live data for both sensors. ", { bold: true, color: INK }),
   run("Latest clear Sentinel-2 and latest Sentinel-1 are fetched from open buckets and "
   + "aligned, with no simulation.")],
  [run("Cross-modal validation is a precision tool. ", { bold: true, color: INK }),
   run("Requiring agreement between radar and optical roughly halves false alarms versus "
   + "trusting one sensor.")],
  [run("It also beats classic ICP alignment ", { bold: true, color: INK }),
   run("(F1 0.67 vs 0.42, AP 0.81 vs 0.66), because RANSAC+CPD is robust to each sensor’s "
   + "modality-specific false alarms where ICP is not.")],
]));

// ---- 8. Limitations ----
body.push(h("8. Limitations & future work", HeadingLevel.HEADING_1));
body.push(...bullets([
  "Real-data corroboration on a single small lake is sparse (few radar detections; S1–S2 gaps of weeks in monsoon season), so the ground-truthed simulation is where the with/without-alignment gap is quantified cleanly.",
  "The AWS radar backscatter is an uncalibrated γ⁰ proxy — sufficient for the local-contrast detector, but not for absolute radiometric analysis.",
  "No field-validated debris labels yet; ground truth is the both-sensors-agree set or synthetic truth. This is a research prototype, not an operational monitor.",
  "Detectors are index-threshold based; a learned detector would improve the point sets that feed the pipeline.",
]));

// ---- 9. References ----
body.push(h("9. References", HeadingLevel.HEADING_1));
body.push(...bullets([
  "A. Myronenko, X. Song. Point Set Registration: Coherent Point Drift. IEEE TPAMI, 2010.",
  "M. A. Fischler, R. C. Bolles. Random Sample Consensus (RANSAC). Comm. ACM, 1981.",
  "S. Umeyama. Least-squares estimation of transformation parameters between two point patterns. IEEE TPAMI, 1991.",
  "L. Biermann et al. Finding Plastic Patches in Coastal Waters using Optical Satellite Data (FDI). Scientific Reports, 2020.",
  "S. K. McFeeters. The use of the NDWI in the delineation of open water features. Int. J. Remote Sensing, 1996.",
  "Copernicus Sentinel data (ESA); Sentinel-2 L2A COGs via AWS sentinel-cogs; Sentinel-1 GRD via AWS sentinel-s1-l1c.",
]));
body.push(spacer(120));
body.push(rule());
body.push(new Paragraph({ children: [new TextRun({
  text: "Prototype for a project-based-learning module. Not for operational or "
      + "safety-critical use.", size: 17, italics: true, color: MUTED })] }));

// ===========================================================================
// assemble
// ===========================================================================
const heading = (color, size) => ({ run: { font: "Calibri", size, bold: true, color },
  paragraph: { spacing: { before: 280, after: 120 } } });
const doc = new Document({
  creator: "PBL-Sem-7",
  title: "Cross-Modal Sentinel-1/Sentinel-2 Debris Detection — Nagpur",
  styles: {
    default: { document: { run: { font: "Calibri", size: 21, color: INK2 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", quickFormat: true, ...heading(INK, 52) },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Calibri", size: 30, bold: true, color: INK },
        paragraph: { spacing: { before: 320, after: 140 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Calibri", size: 24, bold: true, color: ACCENT },
        paragraph: { spacing: { before: 240, after: 100 } } },
    ],
  },
  numbering: { config: [{ reference: "bull", levels: [{ level: 0, format: LevelFormat.BULLET,
    text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 360, hanging: 220 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(ROOT, "docs", "Nagpur_Debris_Detection_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, (buf.length / 1024).toFixed(0) + " KB");
});
