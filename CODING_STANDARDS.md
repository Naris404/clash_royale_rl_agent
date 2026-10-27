# Coding standards

Judgement calls the reviewer enforces. Mechanical rules and the project conventions stay in `AGENTS.md`.

## Reuse the sibling

Before writing a plan driver, training wrapper, ship or report helper, find the module that already does the same job (`experiments/final_configuration.py`, `experiments/arms.py`, `training/report.py`, `training/selection.py`) and extend or call it. A second copy with a slightly different manifest or path layout is a defect: the next change has to be made twice.

## Name each repeated file or directory once

A file or directory name used in more than one module (the final model file, the checkpoints directory, `selection.json`) is a named constant in the module that owns it. Importers use the constant; they do not retype the literal.

## Keep library modules light

A read-only summary or results tool imports plain data types from a light module, never from the training, experiment or server code that produces them. Shared types live where both sides can import them without pulling in `sb3_contrib` or the simulation.

## Keep drivers short

A plan driver reads as a sequence of named steps (train, select, evaluate, ship). When one function does several of them inline, split it by step.
