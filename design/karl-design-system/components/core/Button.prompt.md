Primary interactive control for actions across KARL surfaces — chat send, onboarding steps, admin table actions.

```jsx
<Button variant="primary" size="md" onClick={handleSend}>Send</Button>
<Button variant="gold" size="lg">Start Orientation</Button>
<Button variant="secondary">Cancel</Button>
<Button variant="ghost">Skip</Button>
<Button variant="danger">Delete my data</Button>
```

Variants: `primary` (union blue solid), `gold` (accent solid, for the highest-emphasis single action per screen), `secondary` (white/bordered), `ghost` (transparent, tinted hover), `danger` (destructive red). Sizes: `sm` / `md` / `lg`. Pass `icon` for a leading icon node.
