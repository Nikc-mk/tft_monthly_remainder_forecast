# 2026-04-05 Darts-First MVP Runtime Design

## 1. Objective

Strengthen the project documentation so the MVP runtime contract explicitly requires a simple, readable, research-oriented implementation style for `pipeline/`.

The new documentation must make it unambiguous that the runtime should prefer mature library abstractions over custom infrastructure, with `darts` as the default forecasting library and `pandas`, `numpy`, `pytorch`, and `seaborn` as the preferred supporting stack.

## 2. Problem Statement

The current repository contract already defines the weekly `City` forecasting task and the public runtime boundaries, but it does not yet state strongly enough how the MVP runtime should be implemented.

That leaves room for an overly custom `pipeline/` implementation with:
- handcrafted orchestration where stable library abstractions already exist
- extra framework-like layers that reduce readability
- runtime code that is harder to inspect than the research notebooks
- missing or inconsistent `docstring` coverage in public runtime code

For an exploratory MVP, that direction is counter to the intended style of the project.

## 3. Design Decision

Add an explicit "darts-first" implementation policy to both `docs/technical-assignment.md` and `docs/project-spec.md`.

This policy establishes that:
- `pipeline/` is an MVP runtime, not a custom forecasting framework
- the preferred library stack is `darts`, `pandas`, `numpy`, `pytorch`, `seaborn`
- for forecasting, backtesting, time-series transforms, and related runtime orchestration, the implementation must use `darts` by default when `darts` covers the task
- custom low-level implementations are allowed only as exceptions and must be justified in documentation
- public runtime modules, classes, and functions in `pipeline/` must include meaningful `docstring`

## 4. Documentation Placement

### 4.1 `docs/technical-assignment.md`

This document should carry the practical implementation rule for the MVP runtime.

Required additions:
- a dedicated subsection describing implementation principles for `pipeline/`
- a hard requirement that the runtime remain simple, readable, and short enough for MVP research work
- an explicit statement that the preferred stack is `darts`, `pandas`, `numpy`, `pytorch`, `seaborn`
- an explicit statement that `darts` is the default choice for forecasting and time-series runtime logic when it supports the needed behavior
- an exception rule stating that custom implementations are acceptable only when the required behavior is not adequately provided by the preferred libraries, and that such a choice must be briefly justified in project documentation
- a quality requirement that public runtime code in `pipeline/` must have `docstring`

The document should also point to `notebooks/tft_darts_pipeline.ipynb` as a reference for expected simplicity and style, while keeping the existing rule that notebooks are research-only and are not runtime sources.

### 4.2 `docs/project-spec.md`

This document should carry the normative contract version of the same rule.

Required additions:
- include the policy in `Непереговорные правила`
- include the policy again in `Definition of Done`
- require `docstring` as part of runtime code quality

The wording in this file should make it impossible to interpret the library preference as optional guidance.

## 5. Policy Wording Requirements

The final wording in both documents should enforce the following meanings:
- the runtime must prefer mature, well-documented library abstractions over custom implementations
- `darts` is the primary library for forecasting runtime behavior in this repository
- `pandas`, `numpy`, `pytorch`, and `seaborn` are the preferred supporting libraries
- implementing custom forecasting or backtesting infrastructure from scratch is a deviation, not the default path
- any such deviation requires a short written justification in documentation
- readability and simplicity are mandatory MVP properties, not aspirational goals
- missing `docstring` on public runtime APIs is a quality violation

## 6. Boundaries and Non-Goals

This design changes documentation policy only.

It does not:
- change the weekly `City` forecasting contract
- change the `8`-week horizon, `60`-week history window, or `6`-window rolling backtest
- move runtime logic into notebooks
- require immediate code refactoring inside `pipeline/` as part of the documentation-only change
- ban non-core utility libraries needed for testing, configuration, or packaging

## 7. Reference Usage

`notebooks/tft_darts_pipeline.ipynb` should be referenced as an implementation-style example, not as an executable runtime contract.

That reference is intended to communicate:
- use library capabilities before building custom layers
- keep data flow inspectable
- keep training and inference code direct and readable
- prefer small, well-named helpers with clear `docstring`

## 8. Risks and Mitigations

Risk: the new wording may be interpreted as banning every library outside the preferred stack.
Mitigation: state clearly that the policy is about the default modeling and runtime approach, not a total ban on reasonable supporting utilities.

Risk: the policy may still be written too softly and treated as a recommendation.
Mitigation: place it inside `Непереговорные правила` and `Definition of Done`, and use mandatory wording such as "must" and "required".

Risk: the notebook reference may be misunderstood as permission to keep runtime logic in notebooks.
Mitigation: repeat that notebooks remain research-only and cannot replace `pipeline/`.

## 9. Acceptance Criteria

This design is satisfied when:
- `docs/technical-assignment.md` explicitly states the darts-first MVP runtime policy
- `docs/project-spec.md` explicitly states the same policy as a binding contract
- both documents require simple and readable runtime code in `pipeline/`
- both documents require `docstring` for public runtime code
- both documents allow custom implementations only with explicit written justification
- both documents reference `notebooks/tft_darts_pipeline.ipynb` only as a style reference and not as runtime logic
