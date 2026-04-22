---
description: Update the ARCHITECTURE.md file to be highly visual, using Mermaid diagrams sourced from docs/moon/.
---

You are a System Architect and Technical Writer responsible for maintaining the WSmart+ Route visual architecture documentation. Your goal is to create a highly visual, diagram-centric architecture document with minimal text.

## Context
The application's architectural diagrams are maintained as separate Mermaid (`.mmd`) files in the `docs/moon/` directory. The main `ARCHITECTURE.md` file must serve as the central visual hub, aggregating these diagrams to provide a clear, top-down view of the system.

## Implementation Steps

### 1. Analyze Source Diagrams
- Read and parse all `.mmd` files located in the `docs/moon/` directory.
- Identify the core components, their relationships, and the execution flow represented in these diagrams.

### 2. Structure the Architecture File
- Open or create `docs/ARCHITECTURE.md`.
- Organize the document hierarchically (e.g., High-Level Overview, Data Pipeline, Policy Execution, Evaluation).

### 3. Embed and Render Diagrams
- Embed the content of the `.mmd` files directly into `ARCHITECTURE.md` using Mermaid code blocks:
  ````markdown
  ```mermaid
  <diagram content>
  ```
  ````

- Ensure the syntax is perfectly valid so it renders correctly in standard Markdown viewers.

### 4. Add Minimal Connective Text
- Add very brief, concise headings or single-sentence descriptions above each diagram to explain what it represents.
- **Remove** any long paragraphs, redundant explanations, or deep technical text. Let the diagrams do the talking.

## Guardrails
- **Visuals First:** The file must consist of at least 80% diagrams and no more than 20% text.
- **Single Source of Truth:** Do not invent new architectural flows; strictly translate what is defined in `docs/moon/`.
- **Valid Syntax:** Always verify that the embedded Mermaid syntax is complete and properly enclosed in code blocks.
