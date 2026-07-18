Two-up selectable card for binary classification choices in onboarding (Full-time / Part-time; Member / Steward, etc).

```jsx
<div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:8}}>
  <EmploymentOption title="Full-time" subtitle="35+ hrs/week" selected onClick={...} />
  <EmploymentOption title="Part-time" subtitle="Under 35 hrs/week" onClick={...} />
</div>
```
