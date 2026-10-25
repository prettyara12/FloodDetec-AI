# Design Direction & UI Specification
## FloodDetec AI: Deteksi Banjir Berbasis AI — Studio Console

## 1. Identity & Character
- **Product Name**: FloodDetec AI (Deteksi Banjir Berbasis AI)
- **Domain**: Computer Vision, Semantic Segmentation, AI Flood Detection
- **Visual Personality**: Modern Analytical Studio Workbench. Clean, engineered, high-contrast, and data-dense. Zero decorative neon glows, zero fluffy floating drop-shadows, zero generic AI pill-badges, and zero template layouts.
- **Craftsmanship Standard**: Utilitarian clarity, crisp 1px hairline borders, legible typography, purposeful hierarchy, and comprehensive state handling (empty, loading, active, error).

---

## 2. Anti-Slop Dials (R-37)
- **ENERGY**: 2 (Controlled, authoritative, data-centric)
- **RHYTHM**: 2 (Structured workbench with distinct telemetry and viewport zones)
- **MOTION**: 1 (Immediate 120ms functional transitions for hover and state changes; zero endless pulsing loops or floating effects)

---

## 3. Color Palette & Hierarchy (R-01, R-29)
- **Core Neutral Base (Dark Mode - Default)**:
  - Background Base: `#0B0F17` (Deep Obsidian Slate)
  - Surface Panel / Card: `#111827` (Charcoal Slate)
  - Elevated Surface / Toolbar: `#1F2937` (Muted Slate)
  - Hover Surface: `#283548`
  - Border Hairline: `#374151` / `rgba(255, 255, 255, 0.08)` (Crisp 1px solid, matte, no glow)
  - Border Active / Focus: `#38BDF8` (Sky-400 focus outline, 2px solid, offset 2px)
- **Light Mode Adaptive Alternative (Clean Technical Slate)**:
  - Background Base: `#F8FAFC` (Slate-50)
  - Surface Panel / Card: `#FFFFFF` (Pure White)
  - Elevated Surface / Toolbar: `#F1F5F9` (Slate-100)
  - Hover Surface: `#E2E8F0`
  - Border Hairline: `#CBD5E1` (Slate-300)
  - Border Active / Focus: `#0284C7` (Sky-600)
- **Typography & Contrast Standards (WCAG AA / AAA)**:
  - Text Primary: `#F9FAFB` (Dark) / `#0F172A` (Light) - Contrast ratio > 14:1
  - Text Secondary: `#9CA3AF` (Dark) / `#475569` (Light) - Contrast ratio > 5.5:1
  - Text Muted: `#6B7280` (Dark) / `#64748B` (Light) - Contrast ratio > 4.5:1
- **Domain Semantic Accents (Hydrological Risk Levels)**:
  - Water Mask Overlay: `#06B6D4` (Cyan-500 at 45% alpha) with solid cyan boundary contours for water segmentation.
  - Safe Level (<5% area): `#059669` (Forest Emerald)
  - Low / Waspada (5% to 20%): `#2563EB` (Cobalt Blue)
  - Medium / Siaga (20% to 50%): `#D97706` (Amber Ochre)
  - High / Awas (>50%): `#DC2626` (Crimson Red)

---

## 4. Typography (R-06)
- **Primary Typeface**: `Inter`, `-apple-system`, `BlinkMacSystemFont`, `Segoe UI`, Roboto, sans-serif.
  - Balanced x-height, optimal micro-readability for analytical tables and control labels.
- **Monospace Readout Typeface**: `JetBrains Mono`, `Consolas`, monospace (for numeric coordinates, percentages, latency, and tensor dimensions).

---

## 5. Component Patterns & Anti-Slop Discipline
- **Header**: Clean, slim utility bar with unambiguous navigation and engine status indicator (steady matte dot, no pulsing animation).
- **Studio Workbench**:
  - Integrated top bar with compact drag-and-drop zone and categorized flood sample chips.
  - Unified inspection canvas toolbar: segmented button group for view modes (`Split`, `Overlay`, `Masker Biner`, `Foto Asli`), numeric threshold slider with precise step labels, and direct download action.
  - Precise split comparison slider with draggable handle and keyboard arrow control (`Foto Asli` vs `Deteksi Banjir AI`).
  - Analytical KPI panel: clean tabular metrics (Risk Level, Area %, Pixel Count, Latency) formatted as telemetry readouts instead of marketing cards.
- **Complete UI States (R-27)**:
  - Empty: informative prompt with sample recommendations.
  - Loading: clean linear progress indicator with explicit stage label.
  - Error: clear explanation and retry button.
- **Zero AI-Slop Features**:
  - No decorative emojis in copy or headings (R-04).
  - No eyebrow badge capsules above H1 (R-09).
  - No generic "How It Works 3-Step" templates (R-05).
  - No em dashes in agent copy (R-02).
  - No fake stats or fabricated claims (R-17, R-36).
