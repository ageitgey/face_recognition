# ADR-001 - Local modular monolith with dependency-free browser modules

Status: accepted for S0 boilerplate
Date: 2026-08-16

## Context

Face2AI needs a fast, visible product loop around the existing Python face-recognition library while preserving a clean extraction path for future Party Mirror and agent integrations.

## Decision

Use one local FastAPI process with explicit domain/port/adapter/service boundaries and a same-origin browser UI built from HTML, CSS and ES modules. The UI uses adapted ReactBits-inspired motion patterns without taking a React/Vite dependency in S0.

## Alternatives

1. React + Vite frontend plus FastAPI backend.
2. Tauri/desktop shell plus React and Python sidecar.

## Rationale

The chosen design has one runtime, no frontend package manager, same-origin camera/API flow, and keeps recognition behind an adapter. It minimizes setup risk while allowing later extraction.

## Counterargument

As Party Mirror and richer agent surfaces grow, hand-managed browser state can become harder to evolve than React components. This is a real architectural trade-off.

## Review triggers

Reconsider React/Vite when any two become true:
- more than three independent interactive product surfaces share state;
- reusable component composition becomes a recurring source of duplication;
- frontend automated component testing becomes a bottleneck;
- Party Mirror requires complex timelines, routing, or plugin UI composition.

## Consequences

- No Node build is required for S0.
- UI effects must remain dependency-free and accessible.
- Backend domain boundaries must not depend on HTTP or `face_recognition` directly.
- The recognition adapter can be replaced without changing API/domain contracts.
