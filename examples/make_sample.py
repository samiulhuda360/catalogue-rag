"""Generate the fictional sample catalogues and their evaluation questions.

    python examples/make_sample.py

Writes eleven invented "Acme" catalogues to examples/parsed/ in the same Markdown layout the
ingestion step produces from real PDFs (metadata header, `---` between pages, HTML tables),
plus examples/questions.jsonl. Every product, code and number is made up, and the questions
are generated from the same data, so their expected answers are right by construction.
The output is deterministic and committed, so this only needs re-running after a change.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "parsed"
rng = random.Random(7)

FINISHES = [("Satin Chrome", "SC"), ("Polished Brass", "PB"), ("Matt Black", "MB"), ("Satin Stainless", "SS"), ("Antique Bronze", "AB")]
docs: dict[str, tuple[str, list[str]]] = {}   # doc id -> (title, pages)
questions: list[dict] = []


def table(rows: list[list], head: list[str] | None = None) -> str:
    out = ["<table>"]
    if head:
        out.append("<tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>")
    out += ["<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows]
    return "\n".join(out + ["</table>"])


def bullets(items: list[str]) -> str:
    return "\n".join(f"* {i}" for i in items)


def ask(category: str, question: str, expect: list[list[str]], doc: str) -> None:
    questions.append({"id": f"s{len(questions) + 1:02d}", "category": category, "question": question,
                      "expect": expect, "docs": [doc] if doc else []})


def catalogue(doc: str, title: str, intro: str, pages: list[str]) -> None:
    cover = f"# {title}\n\nAcme Hardware (fictional sample catalogue)\n\n{intro}\n\nEvery product, code and figure in this catalogue is invented."
    docs[doc] = (title, [cover, *pages])


def ordering(code: str, desc: str, finishes: int = 3, hands: bool = False) -> tuple[str, list[tuple[str, str]]]:
    rows, parts = [], []
    for name, fc in rng.sample(FINISHES, finishes):
        for hand in (["LH", "RH"] if hands else [""]):
            part = f"{code}{fc}{hand}"
            rows.append([desc, name] + ([("Left hand" if hand == "LH" else "Right hand")] if hands else []) + [part])
            parts.append((name + (" left hand" if hand == "LH" else " right hand" if hand else ""), part))
    head = ["Description", "Finish"] + (["Handing"] if hands else []) + ["Part number"]
    return "### Ordering information\n\n" + table(rows, head), parts


# ---------------------------------------------------------------- 1. electronic access
pages = []
for model, kind, where in [("H200", "Wireless access handle", "interior doors"), ("H300", "Outdoor wireless handle", "exterior doors"),
                           ("E450", "Stand-alone escutcheon", "interior doors"), ("K60", "Keypad reader", "entrances and gates"),
                           ("R120", "Smart card reader", "wall mounting beside a door")]:
    battery = rng.choice(["1 x Lithium CR123A", "2 x Lithium CR2", "3 x LR03 AAA 1.5V", "4 x LR6 AA 1.5V"])
    months, per_day = rng.choice([24, 30, 36, 40, 48]), rng.choice([10, 20, 25, 30])
    ip = rng.choice(["IP42", "IP54", "IP55", "IP65"]) if model != "H300" else "IP66"
    t_lo, t_hi = rng.choice([(-10, 50), (0, 55), (-20, 60), (-25, 70)])
    thick = rng.choice(["35-60mm", "38-70mm", "40-80mm", "45-90mm"])
    radio = rng.choice(["IEEE 802.15.4 (2.4 GHz)", "Bluetooth Low Energy 5.0", "NFC 13.56 MHz"])
    tech = table([["Battery", battery], ["Battery life", f"{per_day} openings per day<br/>for {months} months"], ["Radio", radio],
                  ["Class of protection", ip], ["Door thickness", thick], ["Operating temperature", f"{t_lo}°C to {t_hi}°C"],
                  ["Credentials", rng.choice(["MIFARE DESFire EV2, PIN", "MIFARE Classic, iCLASS", "PIN, mobile key"])]])
    order, parts = ordering(model, kind, 3, hands=model.startswith("H"))
    pages.append(f"## Acme {model} {kind}\n\n" + bullets([f"Battery-powered {kind.lower()} for {where}",
                 "Online or offline mode, selected during set-up", f"Audit trail of the last {rng.choice([1000, 2000, 5000])} events"])
                 + f"\n\n### Technical data\n\n{tech}\n\n{order}")
    if model == "H200":
        ask("spec", "What battery does the Acme H200 wireless access handle use, and how long does it last?",
            [[next(t for t in ("CR123A", "CR2", "LR03", "LR6") if t in battery)], [f"{months}months", f"{months}month"]], "acme-access-control")
        ask("spec", "Which door thickness does the H200 handle suit?", [[thick, thick.replace("-", " to ")]], "acme-access-control")
        ask("part_number", f"Part number for the H200 handle, {parts[1][0]}?", [[parts[1][1]]], "acme-access-control")
    if model == "H300":
        ask("spec", "What IP rating does the Acme H300 outdoor handle have?", [["ip66"]], "acme-access-control")
    if model == "K60":
        ask("spec", "What is the operating temperature range of the K60 keypad?",
            [[f"{t_lo}°cto{t_hi}°c", f"{t_lo}to{t_hi}°c", f"{t_lo}°c-{t_hi}°c"]], "acme-access-control")
catalogue("acme-access-control", "Acme Electronic Access Control", "Battery-powered handles, escutcheons and readers that work with or "
          "without a network.", pages)

# ---------------------------------------------------------------- 2. door closers
pages = []
for series, kind, sizes in [("C400", "Surface mounted closer", (1, 4)), ("C600", "Heavy duty surface closer", (2, 6)),
                            ("C800", "Fire door closer", (3, 6)), ("F900", "Floor spring closer", (3, 7)), ("CC200", "Concealed closer", (1, 4))]:
    rows = []
    for size in range(sizes[0], sizes[1] + 1):
        rows.append([[750, 850, 950, 1100, 1250, 1400, 1600][size - 1], [20, 40, 60, 80, 100, 120, 160][size - 1], size])
    sizes_t = table(rows, ["Recommended door width (mm)", "Door weight (kg) max", "Closer power size"])
    order, parts = ordering(series, kind, 2)
    pages.append(f"## Acme {series} {kind}\n\n" + bullets(["Adjustable closing speed, latch speed and back check",
                 f"Power sizes {sizes[0]} to {sizes[1]}", rng.choice(["Delayed action option", "Hold-open arm option", "Slim 52mm body"])])
                 + f"\n\n### Selecting the power size\n\n{sizes_t}\n\n{order}")
    if series == "C600":
        r = next(r for r in rows if r[0] == 1100)
        ask("table", "Which C600 closer power size suits an 1100mm door weighing up to 80kg?", [[f"size{r[2]}", f"powersize{r[2]}", f"en{r[2]}"]],
            "acme-door-closers")
    if series == "CC200":
        ask("part_number", f"Part number for the CC200 concealed closer in {parts[0][0]}?", [[parts[0][1]]], "acme-door-closers")
catalogue("acme-door-closers", "Acme Door Closers", "Surface, concealed and floor closers for light to heavy doors.", pages)

# ---------------------------------------------------------------- 3. mortice locks
pages = []
for series, use in [("M52", "commercial entrance doors"), ("M55", "fire-rated doors"), ("M60", "sliding doors"), ("M70", "residential front doors")]:
    backsets = sorted(rng.sample([45, 54, 60, 70, 89, 127], 3))
    funcs = rng.sample(["Entrance", "Classroom", "Office", "Privacy", "Storeroom", "Passage", "Escape"], 4)
    rows = [[f"{series}{i + 1:02d}", f, f"{backsets[i % 3]}mm"] for i, f in enumerate(funcs)]
    order, parts = ordering(series, f"{series} lock case", 3)
    pages.append(f"## Acme {series} Series Mortice Lock\n\nFor {use}. Available backsets: {', '.join(f'{b}mm' for b in backsets)}.\n\n"
                 + bullets([f"Latch throw {rng.choice([12, 14, 16])}mm, deadbolt throw {rng.choice([20, 25, 28])}mm",
                            "Reversible latch without opening the case", f"Fire rated to {rng.choice([60, 120, 240])} minutes"])
                 + f"\n\n### Functions\n\n{table(rows, ['Model', 'Function', 'Backset'])}\n\n{order}")
    if series == "M52":
        ask("spec", "Which backsets does the M52 series mortice lock come in?", [[f"{b}mm", str(b)] for b in backsets], "acme-mortice-locks")
    if series == "M55":
        ask("table", f"Which M55 model has the {funcs[2]} function?", [[f"{series}03"]], "acme-mortice-locks")
catalogue("acme-mortice-locks", "Acme Mortice Locks", "Mortice lock cases for commercial, fire-rated, sliding and residential doors.", pages)

# ---------------------------------------------------------------- 4. padlocks and chains
rows, chain = [], []
for shackle, body in [(6, 30), (8, 40), (10, 50), (11, 60), (13, 70)]:
    rows.append([f"P{body}", f"{body}mm", f"{shackle}mm", f"{rng.choice([20, 25, 30, 60])}mm", rng.choice(["Brass", "Hardened steel", "Stainless"])])
for dia, length in [(6, 600), (8, 900), (10, 900), (10, 1200), (13, 1500)]:
    chain.append([f"CH/{dia:02d}/{length}", f"{dia}mm dia x {length}mm long hardened steel chain"])
catalogue("acme-padlocks", "Acme Padlocks and Chains", "Padlocks in five body sizes and hardened steel security chains.", [
    "## Padlock range\n\n" + table(rows, ["Model", "Body width", "Shackle diameter", "Shackle clearance", "Body"]),
    "## Security chains\n\n" + table(chain, ["Part number", "Description"]),
    *[f"## Acme {r[0]} padlock\n\n" + bullets([f"{r[1]} {r[4].lower()} body, {r[2]} shackle", f"Shackle clearance {r[3]}",
      rng.choice(["Keyed alike or keyed to differ", "Key retaining: the key cannot be removed while open", "Weatherproof shutter over the keyway"]),
      f"Supplied with {rng.choice([2, 3, 4])} keys"]) for r in rows],
])
ask("table", "What is the shackle diameter of the Acme P50 padlock?", [["10mm"]], "acme-padlocks")
ask("part_number", "Part number for the 10mm x 1200mm hardened steel chain?", [["ch/10/1200"]], "acme-padlocks")

# ---------------------------------------------------------------- 5. window hardware
stays = [[f"WS{h}", f"{h}mm", f"{w}mm", f"{kg}kg"] for h, w, kg in [(200, 450, 8), (250, 600, 12), (300, 750, 16), (400, 900, 22), (500, 1100, 30)]]
winders = [[f"WW{n}", f"{n}mm", rng.choice(["Awning", "Casement"]), f"{rng.choice([20, 25, 35])}kg"] for n in (200, 300, 400)]
catalogue("acme-window-hardware", "Acme Window Hardware", "Friction stays, winders and fasteners for timber and aluminium windows.", [
    "## Friction stays\n\nChoose the stay length from the sash height.\n\n"
    + table(stays, ["Part number", "Stay length", "Max sash height", "Max sash weight"]),
    "## Window winders\n\n" + table(winders, ["Part number", "Arm length", "Window type", "Max sash weight"]),
    "## Wedgeless fasteners\n\n" + bullets(["Locking and non-locking versions", "Suits sashes from 15 to 30mm", "Key: Acme W1 universal window key"]),
])
ask("table", "Which friction stay suits a 750mm sash height?", [["ws300"]], "acme-window-hardware")

# ---------------------------------------------------------------- 6. sliding door gear
rollers = [[f"SR{kg}", f"{kg}kg", rng.choice(["Single", "Tandem"]), rng.choice(["Nylon", "Steel ball bearing"])] for kg in (40, 60, 80, 120, 160)]
catalogue("acme-sliding-doors", "Acme Sliding Door Gear", "Top-hung and bottom-rolling gear for interior and exterior sliding doors.", [
    "## Bottom rollers\n\n" + table(rollers, ["Part number", "Max door weight", "Wheel", "Bearing"]),
    *[f"## {r[0]} bottom roller\n\n" + bullets([f"Carries doors up to {r[1]}", f"{r[2]} wheel, {r[3].lower()}",
      f"Height adjustment {rng.choice([6, 8, 10])}mm", "Suits 4mm upstand track"]) for r in rollers],
    "## Top-hung track\n\n" + table([[f"TT{n}", f"{n}mm", "Aluminium"] for n in (1800, 2400, 3000, 3600)], ["Part number", "Length", "Material"]),
])
ask("table", "Which bottom roller carries a 120kg sliding door?", [["sr120"]], "acme-sliding-doors")

# ---------------------------------------------------------------- 7. safes
rows = []
for name, litres in [("Home", 16), ("Office", 38), ("Fire", 45), ("Commercial", 90), ("Vault", 210)]:
    rows.append([f"Acme {name} Safe", f"S{litres}", f"{litres} litres", f"{round(litres * rng.uniform(0.9, 1.6))}kg",
                 {"Home": "None", "Office": "30 minutes", "Fire": "120 minutes", "Commercial": "60 minutes", "Vault": "120 minutes"}[name],
                 rng.choice(["Electronic keypad", "Key", "Dual keypad and key"])])
safes_t = table(rows, ["Model", "Part number", "Capacity", "Weight", "Fire protection", "Lock"])
safe_pages = [f"## {r[0]}\n\n" + bullets([f"Capacity {r[2]}, weight {r[3]}", f"Fire protection: {r[4]}", f"Lock: {r[5]}",
               f"{rng.choice([2, 4, 6])} anchor bolts included", f"{rng.choice([1, 2, 3])} adjustable shelves"]) for r in rows]
catalogue("acme-safes", "Acme Safes", "Safes for homes, offices and businesses.", ["## Safe range\n\n" + safes_t, *safe_pages])
ask("table", "How heavy is the Acme Commercial Safe?", [[rows[3][3]]], "acme-safes")

# ---------------------------------------------------------------- 8. exit devices
rows = [[f"X{w}", f"{w}mm", rng.choice(["Rim", "Mortice", "Vertical rod"]), rng.choice(["Yes", "No"])] for w in (900, 1000, 1100, 1200)]
catalogue("acme-exit-devices", "Acme Exit Devices", "Push bars for escape doors.", [
    "## Push bar exit devices\n\n" + table(rows, ["Part number", "Door width", "Type", "Fire rated"]),
    *[f"## {r[0]} {r[2].lower()} exit device\n\n" + bullets([f"Suits doors up to {r[1]} wide", f"Fire rated: {r[3]}",
      f"Dogging by {rng.choice(['hex key', 'cylinder'])}", f"Projection {rng.choice([70, 85, 95])}mm"]) for r in rows],
])

# ---------------------------------------------------------------- 9. hinges and bolts
rows = [[f"HB{s}", f"{s}x{s - 25}mm", f"{kg}kg", "Stainless steel"] for s, kg in ((100, 40), (113, 80), (125, 120))]
bolts = [[f"FB{n}", f"{n}mm", rng.choice(["Lever action", "Slide action"])] for n in (150, 200, 300, 450)]
catalogue("acme-hinges-bolts", "Acme Hinges and Bolts", "Ball-bearing hinges and flush bolts.", [
    "## Ball-bearing butt hinges\n\nLoad is per set of three hinges.\n\n" + table(rows, ["Part number", "Size", "Load per set", "Material"]),
    "## Flush bolts\n\n" + table(bolts, ["Part number", "Length", "Action"]),
])
ask("table", "What load can a set of three HB113 hinges carry?", [["80kg"]], "acme-hinges-bolts")

# ---------------------------------------------------------------- 10. keying
rows = [[f"K{a}/{b}", f"{a}/{b}mm", f"{a + b}mm", rng.choice(["5 pin", "6 pin"])] for a, b in ((30, 30), (30, 40), (35, 45), (40, 50))]
catalogue("acme-keying", "Acme Keying and Cylinders", "Euro profile cylinders and master-key systems.", [
    "## Euro profile double cylinders\n\n" + table(rows, ["Part number", "Sizes (A/B)", "Overall length", "Pins"]),
    "## Master-key systems\n\n" + bullets(["Up to 3 levels: grand master, master, change keys",
                                            "Registered key profile: keys cut only on proof of ownership"]),
])
ask("part_number", "Part number for a 35/45 euro profile double cylinder?", [["k35/45"]], "acme-keying")

# ---------------------------------------------------------------- 11. lever handles (fixed data: no random draws,
# so adding it leaves every other catalogue unchanged)
levers = [["L10", "Residential lever set", "Homes: interior and entry doors", "Light duty", "No", "No"],
          ["L20", "Commercial lever set", "Offices, schools, retail", "Heavy duty, 1,000,000 cycles", "Yes, 120 minutes", "No"],
          ["L30", "Accessible lever set", "Public buildings, healthcare, aged care", "Heavy duty, 1,000,000 cycles", "Yes, 60 minutes", "Yes"],
          ["L40", "Digital lever set", "Homes and small offices", "Medium duty", "No", "No"]]
catalogue("acme-lever-handles", "Acme Lever Handles", "Mechanical and digital lever sets for residential and commercial doors.", [
    "## Lever set range\n\n" + table(levers, ["Model", "Name", "Typical use", "Duty", "Fire rated", "Accessible design"]),
    "## Acme L10 Residential lever set\n\n" + bullets(["For interior and entry doors in homes", "Passage, privacy and entrance functions",
                                                       "Not fire rated: for fire doors use the L20 or L30", "Finishes: Satin Chrome, Matt Black"]),
    "## Acme L20 Commercial lever set\n\n" + bullets(["For offices, schools and retail doors with heavy traffic",
                                                      "Fire rated to 120 minutes when fitted to an M55 lock case",
                                                      "Tested to 1,000,000 operating cycles", "Spring-assisted lever that does not droop"]),
    "## Acme L30 Accessible lever set\n\n" + bullets(["Designed for accessible doors in public buildings, healthcare and aged care",
                                                      "Return-to-door lever end so clothing and bags do not catch",
                                                      "19mm round grip, operable with a closed fist", "Operating torque under 1 Nm",
                                                      "High-contrast Matt Black finish for light-coloured doors",
                                                      "Fire rated to 60 minutes when fitted to an M55 lock case"]),
    "## Acme L40 Digital lever set\n\n" + bullets(["Keypad entry with up to 100 user codes", "Mechanical key override for flat batteries",
                                                   "Auto-relock after 5 seconds", "Mechanical inside lever: always free egress"]),
])
ask("spec", "Which Acme lever set is designed for accessible doors, and what makes it accessible?",
    [["l30"], ["return-to-door", "returntodoor", "closedfist", "19mm"]], "acme-lever-handles")
ask("table", "Which Acme lever set suits commercial offices and is fire rated to 120 minutes?", [["l20"]], "acme-lever-handles")
ask("spec", "Does the L40 digital lever set have a mechanical key override?", [["keyoverride", "mechanicalkey"]], "acme-lever-handles")

# ---------------------------------------------------------------- unanswerable: the right answer is to decline
for q in ["How much does the Acme H200 handle cost?", "Is the Acme M52 lock approved for use with Northgate access control?",
          "Which building code clause covers the C800 fire door closer?", "What is the warranty on the Acme Vault Safe?",
          "What battery does the Acme H900 handle use?", "Is the L30 lever set certified to AS 1428.1?"]:
    ask("unanswerable", q, [], "")

OUT.mkdir(parents=True, exist_ok=True)
for old in OUT.glob("*_content.md"):
    old.unlink()
for doc, (_title, pages) in docs.items():
    header = f"# {doc}.pdf\n\n**Brand:** Acme (fictional sample data)\n**Parsed:** 2026-10-03\n\n---\n\n"
    (OUT / f"{doc}_content.md").write_text(header + "\n\n---\n\n".join(pages) + "\n", encoding="utf-8")
(HERE / "questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions), encoding="utf-8")
print(f"{len(docs)} catalogues, {sum(len(p) for _, p in docs.values())} pages, {len(questions)} questions")
