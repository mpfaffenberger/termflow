# Mermaid Diagram Examples

This file demonstrates termflow's Mermaid diagram rendering capabilities.

## Simple Flowchart

A basic left-to-right flowchart:

```mermaid
graph LR
    A[Start] --> B[Process]
    B --> C[End]
```

## Decision Tree

A top-down decision flowchart with conditional branches:

```mermaid
graph TD
    A[User Request] --> B{Authenticated?}
    B -->|Yes| C[Process Request]
    B -->|No| D[Show Login]
    C --> E[Return Response]
    D --> F[Redirect]
```

## Different Node Shapes

Showcasing all available node shapes:

```mermaid
graph LR
    A[Rectangle] --> B(Rounded)
    B --> C{Diamond}
    C --> D((Circle))
    D --> E>Flag]
```

**Shape Reference:**
- `[text]` - Rectangle
- `(text)` - Rounded rectangle
- `{text}` - Diamond (decision)
- `((text))` - Circle
- `>text]` - Flag/asymmetric

## Edge Styles

Different edge/arrow styles:

```mermaid
graph LR
    A -->|Solid Arrow| B
    B ---|Solid Line| C
    C -.->|Dotted Arrow| D
    D ==>|Thick Arrow| E
```

**Edge Reference:**
- `-->` - Solid line with arrow
- `---` - Solid line without arrow
- `-.->` - Dotted line with arrow
- `==>` - Thick line with arrow

## Right to Left

Graphs can flow in different directions:

```mermaid
graph RL
    End[End] --> Process[Process]
    Process --> Start[Start]
```

## Bottom to Top

```mermaid
graph BT
    C[Top] --> B[Middle]
    B --> A[Bottom]
```

## Complex Example

A more realistic software architecture example:

```mermaid
graph TD
    Client[Browser Client] --> API{API Gateway}
    API -->|Auth| Auth[Auth Service]
    API -->|Data| DB[(Database)]
    API -->|Cache| Cache((Redis))
    Auth --> DB
    DB --> Cache
```

## Git Workflow

```mermaid
graph LR
    A[Working Dir] -->|git add| B[Staging]
    B -->|git commit| C[Local Repo]
    C -->|git push| D[Remote Repo]
    D -->|git pull| A
```

---

*Render this file with: `tf examples/mermaid.md`*
