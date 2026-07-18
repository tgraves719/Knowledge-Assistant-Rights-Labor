Karl's brand mark — a two-half shield colored per-union (left/right halves take the tenant's two brand colors). In the live product it splits apart and reveals an animated scanning-paper motif while Karl is "thinking"; this recreation exposes that as an `idle` vs `thinking` state prop.

```jsx
<ShieldMark size={52} state="idle" />
<ShieldMark size={52} state="thinking" leftColor="#4A7A9F" rightColor="#EECF6D" />
```

Default colors are the shipped UFCW Local 7 tenant (`--union-shield-left/right`). Always recolor per-union — never hardcode blue/gold as if it were KARL's universal brand.
