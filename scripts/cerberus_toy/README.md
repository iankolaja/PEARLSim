# Cerberus pebble-parking toy test

Checks whether a live Serpent session (HxF build, driven through Cerberus) can stand in
for PEARLSim's fresh-Serpent-per-step workflow by parking unused pebbles at radius 0.
Background, design and source citations: `docs/serpent/cerberus_hxf.md` (workspace).

1. `build_toy.py OUT [--scale-pebbles 3000000]` writes the decks (no Serpent; fine on the
   login node).
2. `run_toy.sub` (batch) runs `driver.py`.
   - Reference: ordinary runs of layouts A, B, C.
   - Live session: layout A → B (moved pebbles, new fuel volumes) → back to A → a 6.525-day
     depletion step.
   - Second session: an emptied fuel slot (child volume 1e-6 cm³; Serpent rejects 0).
   - Third session: a 3 M-pebble pool for start-up and send/rebuild timing.

What `results.json` should show if the design works:
- live k equals the reference k within statistics (|z| ≲ 2) for A, B and A after depletion;
- U-235 after depletion close to the reference (the live session's random-number stream differs, so
  only statistical agreement is expected);
- an emptied slot matches its reference;
- start-up time and full-array send time at 3 M pebbles.

Caveat: the scale case has only 8 material universes, so it under-measures Serpent's per-pebble
universe lookup at start-up, which grows with pebbles × universes (~520 universes in a real core).

Live decks differ from the reference decks in two lines, both forced by the HxF build:
no `set inventory` (it crashes the output-variable listing) and `dep daytot` instead of
`dep daystep` (`advance_to_time` supports only cumulative steps). Only divided children
(`<mat>pp<k>r1`) take live volumes. Details: `docs/serpent/cerberus_hxf.md`.

Result (2026-10-16, job 40039974): all checks pass, |z| ≤ 0.5; EOS U-235 within 0.015 %;
live start-up ~45 s at both 4 k and 3 M pebbles; 3 M-pebble send ~4 s.
