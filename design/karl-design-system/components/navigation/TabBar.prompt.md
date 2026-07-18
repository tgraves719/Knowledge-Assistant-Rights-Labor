Primary app navigation — bottom bar on mobile, top bar on desktop. Active tab's icon and label glow gold (drop-shadow + text-shadow), matching the live product exactly.

```jsx
<TabBar
  tabs={[{id:'chat',label:'Chat',icon:'💬'},{id:'contract',label:'Contract',icon:'📄'},{id:'settings',label:'Settings',icon:'⚙️'}]}
  active="chat"
  onChange={setTab}
/>
```
