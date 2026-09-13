# Changelog

## 0.1.0 — 2026-09-13

First release. Margin runs entirely on your own computer, on a CPU or a GPU.

### What it does

- **Install and run with one command each.** The installer sets up an isolated
  Python, the pinned llama.cpp runtime for your hardware, the Gemma 4 E4B model
  and the search models. `margin` opens the study interface.
- **Reads messy material:** PDFs (text and scanned), Word, PowerPoint, and JPEG,
  PNG and WebP photos, with OCR for pages that have no text layer. Two full
  textbooks (2,306 pages) round-trip with 100% of their text kept.
- **Searches your library** with keyword and meaning-based matching combined.
- **Exam blueprint:** reads past papers and ranks the chapters they weigh most.
  In testing it found 93% of the chapters an exam favoured.
- **Notes you can trust:** every sentence is checked against the book, and
  unsupported ones are removed (99.3% of sentences kept). Each section is saved
  as it finishes, so a stopped run resumes where it left off.
- **Diagrams and tables:** flowcharts, tables, mind maps, timelines, formula
  sheets and glossaries, repaired so they always render.
- **Practice:** exam questions that must be answerable from the book, must not
  copy its exercises and must be worth the marks asked for (17 of 18 passed);
  answers marked point by point; key terms exported as an Anki deck.
- **An assistant with 14 tools** that searches, reads, draws, calculates, writes
  questions, marks answers and exports.
- **Exports** Markdown, Anki decks, and HTML files that work offline with the
  diagrams drawn and print cleanly to PDF.
- **Offline, and tested to be:** tests fail if anything tries to reach the
  internet. The whole flow passed 8 of 8 steps from the interface with the
  network blocked.

### Known limits

- Answer and note checks compare words, not meaning; formula-heavy answers are
  the weak spot.
- Math in exports is shown as written, not typeset.
- The blueprint is dependable at chapter level and rough at section level.
- Measurements come from one physics textbook and one GPU machine; CPU speed was
  benchmarked, but the full interface flow has only been run on a GPU.
