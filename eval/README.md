# eval/ - adversarial suites for MC3

`score.py` is the official scorer. `gen_corpus.py` builds corpora and gold files in the exact format of
`sample-questions.json`; `oracle_outputs.py` turns a gold file into harness outputs.

## Generate a suite

```
python3 eval/gen_corpus.py --seed 1 --out eval/suites/s1 [--n-files 40] [--traps T1,T2,T14b,...]   # default: every trap
```

Writes (under `eval/suites/`, which is gitignored):

| Path | Content |
|---|---|
| `s1/corpus/` | the files (default 40 + the v1 traps' files, ~76 with every trap; subdirs, pdf docx xlsx csv log py txt md yaml json png jpg dat + an empty dir) |
| `s1/questions.json` | gold: `challenge, description, corpus, answer_rule, queries[], seed`; each query `n, query, expected_answer, answer_aliases, expected_citations, category, note` (+ `unanswerable: true` for refusals) |
| `s1/traps.json` | `{"<query n>": "<trap id>"}` |
| `s1/MANIFEST.txt` | every file with size, expected `parse.py` status, which file is chmod 000, which PDF is encrypted |

Deterministic: same `--seed` gives byte-identical files and gold (OOXML zips use fixed timestamps, including the `modified` stamp openpyxl writes on save).
Every seed draws a new fictional universe (company, firmware name, product line `KX-72` / `VR-310`,
part numbers `HAL-PSU-0931-C`, tickets `HLX-2041`, error codes `E4410`, versions `7.2.1`, quarters
`Q2 FY28`, pins, serials, FCC ids); small integers can still coincide between seeds.
The generator self-validates the gold and redraws (up to 50 attempts) if an image-only answer leaks into a text
file, an answer is missing from its cited text, or an identifier answer also sits in an uncited file.

## Prove the gold is self-consistent

```
python3 eval/oracle_outputs.py eval/suites/s1/questions.json /tmp/oracle_s1
python3 eval/score.py eval/suites/s1/questions.json /tmp/oracle_s1      # TOTAL 20 x number of queries (v0 traps only: 920/920 for 46)
```

## Measure the pipeline (CPU only)

```
python3 eval/recall.py eval/suites/s3 eval/suites/s4 --ocr          # is every gold file + a record with the value in the package?
python3 eval/run_suite.py --corpus eval/suites/s3/corpus --questions eval/suites/s3/questions.json \
    --out /tmp/s3_oracle --reader oracle                            # verifier/gates/citations with a perfect reader
```

With `--reader oracle` every canary is refused by the reader itself, so only the answerable questions are informative
(every non-OK one is a false refusal or a wrong citation set of the deterministic part). To see what the deterministic gates
alone stop, run `--reader lexical` / `--reader adversarial` (wrong-on-purpose readers) and count `FALSE ANSWERS`.
`recall.py` only tests the canonical gold string: `6.9.1` (source `v6.9.1`) and `-20` (source `U+2212 20`) are checked
through their aliases when you replay a real reader.

## Traps (v0 = T1-T12 and T14, v1 = T13-T24 below, plus the canaries)

Always present (file-level): empty `archive/`; `vendor/internal_audit.txt` with `chmod 000`
(unreadable unless root); `vendor/<supplier>_agreement_ENCRYPTED.pdf` (pypdf, user password `secret`)
holding the price/payment terms/liability cap that canaries ask for; `vendor/telemetry_capture.dat`
(16 KB random bytes); `vendor/supplier_notes.txt` (says pricing is "in the agreement itself").

| Id | Trap | Gold |
|---|---|---|
| T1 | `<prod>_datasheet_r1_WITHDRAWN.pdf` vs `_r2.pdf` | r2 only |
| T2 | r1 withdrawn -> r2 value -> `specs/errata_<prod>.txt` correcting Table 3 | errata only (it alone holds the value and names the field); second Q whose field is untouched -> r2 |
| T3 | xlsx row `status=superseded`, row with `(legacy)` suffix, hidden `Archive` sheet | current row |
| T4 | `.py` comment / module docstring names the old value, literal is the answer | literal; aliases `"<v> seconds"`, `"<v>s"` |
| T5 | near-miss row for the sibling product (fan, cable cost; sibling datasheet) | the named product's row |
| T6 | log `incident logged against <ticket>` -> bug db `fixed_in`; direct and single-hop forms | multi-hop cites log + csv; direct cites csv only |
| T7 | fleet log (`DEV-nnnn`) -> `support/fleet.csv` -> bug db | all three files |
| T8 | board power in datasheet and in `marketing/*_product_brief.txt` | datasheet only, with and without "according to the datasheet" (second is ambiguous by design, noted) |
| T9 | short `support/cooling_quickref.txt` vs long `liquid_cooling_whitepaper.md` with other numbers; FAQ with a typical-temperature decoy; runbook md with sibling step | the short txt / the runbook step for the named product |
| T10 | pin map png (pin of a signal) and asset label jpg (revision, serial, FCC id); values appear in no text file | the image |
| T11 | tape-out quarter only inside a picture embedded in a docx | the docx |
| T12 | roadmap docx with a real Word table | quarter from the table row |
| T14 | `2,500` style number (minimum order quantity) | verbatim with comma; aliases `2500`, `2,500 units` |

### v1 (T13-T24, adversary round 1)

Every v1 trap draws its values from `ctx.xr("<name>")` (a per-trap RNG), so they never depend on which other traps ran.
Not every v1 answer is meant to be parseable by the current app: the `by design` column says what the trap is *expected* to hit.

| Id | Trap | Gold | By design |
|---|---|---|---|
| T13 | `planning/<p2>_rack_bom.xlsx`: BOM sheet with merged title, `total_usd` = formula `=qty*unit` **without cached value**, `#,##0.00` formats; `Quarterly` sheet with merged title, a note on row 2 and its **header on row 3**, `attach_rate` = formula with `0.0%`; hidden `Prior_Estimate` sheet with stale totals and a legacy item | total as displayed (`12,480.00`, aliases `$12,480.00`, `12480.00`, `12480`); percent `34.5%` (alias `34.5`); unit cost with separator | formula evaluator, number formats, merged cells, hidden-sheet staleness |
| T14b | `specs/<p2>_preliminary_spec.docx` (Word table): `$1,299.00`, `12.5%`, `2.4 TB/s`, `U+2212 40 C`, `v4.3.2`, `FY28 Q2` | `1,299.00` (aliases `$1,299.00`, `1299`, `1299.00`), `12.5%` (`12.5`), `2.4`, `-40` (`-40 C`, `U+2212 40`), `4.3.2` (`v4.3.2`), `FY28 Q2` (`Q2 FY28`) | span snapping around prefixes, unicode minus, FY-first quarter |
| T15b | token boundary: `support/escalation_log.csv` (`HLX-204` next to `HLX-2041`/`HLX-2047`), `engineering/bootloader_builds.csv` (`5.0.1` next to `5.0.10..12`), `specs/<p2>_expansion_connector.csv` (`K3` next to `K34`/`K30`); each as "id in the question" and "id is the answer" | the short token's row, verbatim | token-bounded span search |
| T17b | false-premise variants and look-alikes: sibling-only attribute (`<p4>` developer-kit price), nonexistent `r3` (and a nonexistent file name); **answerable** questions that name a real file (`roadmap_fy28.docx`, `<p0>_datasheet_r2.pdf`) or contain `HBM3e`, `PCIe 5.0`, `12V-2x6`, `r2` | refusals / the real value | premise gate must not fire on unindexable-but-real tokens |
| T18b | `specs/smc<n>_spec_r1_WITHDRAWN.pdf`, `_r2.pdf` (name clean, title says SUPERSEDED BY REVISION 3), `_r3.pdf`, `errata_smc<n>.txt` (overrides the watchdog field of r3); `planning/roadmap_fyNN_DRAFT.docx` next to the final (only name + a heading say draft); `procurement/po_register.csv` with a `status=cancelled` row next to the live one | r3 value; errata value (errata only, like T2); r3 value for an untouched field; final roadmap cell; live PO row | title-only supersession, DRAFT, cancelled |
| T19b | `engineering/<p0>_operating_limits.md`: unit asked != unit in source (ms->s, C->F, seconds->minutes, GiB->GB) | refuse; plus `45 secs` asked "in seconds" and `120 ms` asked "in milliseconds" are answered | unit gate |
| T20 | `logs/stress_run_<d>.log` (3,000 lines, answer line ~2,700, 40 decoy ERROR lines), `specs/<p0>_integration_handbook.pdf` (150 pages, value on page 120), `support/field_returns.csv` (5,000 rows, ticket row ~4,850) | the code / cycles / part | `parse.PDF_MAX_PAGES = 60` hides page 120 (parser issue, expected) |
| T21 | `facilities/<site>_site_survey.txt` (UTF-16 + BOM), `support/sensor_calibration.csv` (`;`, CRLF, decimal comma `1,25`), `engineering/power_rails.md` (markdown table), `logs/facility_events_<d>.log` (CRLF), `engineering/agent_config.yaml` (same key under two parents), `engineering/limits.json` (same key under two parents) | the value of the right parent | decoding, nested-key context lost by line records |
| T22 | `planning/Board Review Notes.docx`, `specs/Übersicht_Kühlung.pdf`, `support/café_orders.csv`, `specs/CONNECTOR_SHEET.PDF`, `support/INVENTORY_SUMMARY.XLSX`, `specs/report.final.pdf`, `v2.0/release_manifest.txt`, a 200-character file name | value; citation == exact relative path | paths with spaces / unicode / upper-case extension / dotted dir |
| T23 | `logs/maintenance_<d>.log` -> `support/work_orders.csv` where the bridging work order also sits in `labor_hours.csv` + `ops_review_notes.txt` (df 4) or only in the notes (df 3); 3-hop `support/chassis_label.jpg` (serial only on the picture) -> `chassis_registry.csv` -> `chassis_service.csv` | the part; citations = the necessary files only (log + work_orders; label + registry + service) | rare-id hop with `RARE_DF = 3`, OCR-glued label |
| T24 | `support/shipping_label_rot90.jpg` (rotated), `specs/<p3>_service_pinout.png` (`THERM_ALERT#` vs `THERM_ALERT2#`, staggered label rows), `support/calibration_sticker.jpg` (low contrast), `support/rack_photo.jpg` (noise, blur, tilted plate), `planning/<p4>_power_budget.docx` (answer only in an embedded PNG table) | value read from the picture; none of them appears in any text file | OCR rotation/pairing, VLM |

v1 canaries (tag = source trap + `c`; the in-trap refusals of T17b/T19b keep their own tag): `T13c` hidden-sheet-only item (x2) and arithmetic on a BOM
line; `T14bc` "total price of 3 kits"; `T19c` "total cost of 5 cables"; `T19bc` range values ("between 8 and 14 L/min", fan speed); `T16` unreadable /
encrypted file **named in the question**; `T18bc` product only in the DRAFT roadmap (x2), part only on a cancelled PO row, field only in r1, field only in
r2 (superseded by title); `T20c` event never logged in the 3,000-line run, RMA id absent from the 5,000 rows; `T23c` nonexistent chassis; `T15bc`
ids that are only a prefix/extension of real ones (`HLX-20`, `5.0.100`, `K35`); `T21c` site survey for another site. With every trap on, 48 of the 129 questions
are refusals.

Validation additions: `ctx.forbid` (tokens fillers must never contain), `ctx.absent` (well-formed ids that must be in no file), `ctx.secrets`
(token -> the only files allowed to carry it, e.g. the label serial lives only in the registry), `imgonly=True` on `ctx.q` (answer lives only in a
picture embedded in a docx; short numeric answers are then checked against the cited file only).

v0 canaries (20 per seed, `expected_answer ""`, `unanswerable: true`; trap ids in `traps.json`):
T16 no answer (encrypted pdf x3, unreadable audit file x2, sibling-only attribute x2, "-" in table, event never logged),
T17 false premise (nonexistent product x3, ticket x2, device, error code), T18 value only in the withdrawn r1 (x2),
T19 needs unit conversion / arithmetic not in the corpus (x2). A canary is only emitted when the traps it depends on are generated.

Answer format: value only (`87`, `HAL-PSU-0931-C`, `Q2 FY28`, `7.2.1`, `B14`, `REV-D1`); prices for answerable questions
`84.50` with aliases `$84.50`, `84.5`, `USD 84.50`; versions alias `"<Firmware> 7.2.1"`; pins alias `"pin B14"`.

## Adding a trap

Write `trap_T25(ctx)` in `gen_corpus.py` (see `trap_T13` ... `trap_T24` for the v1 style: values from `ctx.xr(name)`): register files with `ctx.text(path)` (txt/md/py/csv/log lines), `ctx.doc(path, "pdf"|"docx"|"xlsx"|"png"|"jpg"|"bin")`
(or reuse the idempotent `ensure_*` helpers for shared files), draw values in `make_universe` (so they stay stable whatever `--traps` is),
add questions with `ctx.q(trap_id, query, answer, cites, category, note, aliases=..., also=[files that may legitimately repeat the answer])`,
append `"T25"` to `TRAPS`, and add canaries that need it to `canaries(ctx)` / `canaries_x(ctx)`.
Text files can be written as UTF-16 (`doc.encoding = "utf-16"`) or CRLF (`doc.newline = "\r\n"`); xlsx sheets accept formulas (strings starting with `=`),
`pre` rows above the header, `fmt` per column, `shown` (what the sheet displays, for validation) and `hidden`; pdf blocks accept `("page", "")` for a page break.
