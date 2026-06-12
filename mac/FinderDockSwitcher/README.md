# FinderDockSwitcher

Small macOS Dock app that shows your open Finder windows when you right-click its Dock icon and brings the selected window to the front.

## Requirements

- macOS 13 or newer
- Apple command line tools installed

## Build

```bash
cd /Volumes/Seagate/Projects/Codex/UnionKarlUpdated/mac/FinderDockSwitcher
chmod +x scripts/build_app.sh
./scripts/build_app.sh
```

The built app bundle will be created at:

```text
/Volumes/Seagate/Projects/Codex/UnionKarlUpdated/mac/FinderDockSwitcher/dist/FinderDockSwitcher.app
```

## Run

Open the app bundle in Finder, or launch it from Terminal:

```bash
open /Volumes/Seagate/Projects/Codex/UnionKarlUpdated/mac/FinderDockSwitcher/dist/FinderDockSwitcher.app
```

The app stays in the Dock. Right-click its Dock icon to see open Finder windows.

## Permissions

The first time the app talks to Finder, macOS should prompt for Automation permission. Approve access so the app can read Finder windows and bring the selected one forward.

## Notes

- This adds its own Dock icon. macOS does not allow third-party apps to inject custom items into Finder's existing Dock menu.
- The current version keys each window by its folder path. If you keep multiple Finder windows open to the exact same folder, the first matching one will be focused.
