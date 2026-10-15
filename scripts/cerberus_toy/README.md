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
   - Second session: an emptied fuel slot at volume 0.
   - Third session: a 3 M-pebble pool for start-up and send/rebuild timing.

What `results.json` should show if the design works:
- live k equals the reference k within statistics (|z| ≲ 2) for A, B and A after depletion;
- U-235 after depletion close to the reference (the live session's random-number stream differs, so
  only statistical agreement is expected);
- whether an empty slot at volume 0 is accepted;
- start-up time and full-array send time at 3 M pebbles.

Caveat: the scale case has only 8 material universes, so it under-measures Serpent's per-pebble
universe lookup at start-up, which grows with pebbles × universes (~520 universes in a real core).
