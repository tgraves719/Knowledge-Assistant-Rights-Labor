#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PRODUCT_NAME="FinderDockSwitcher"
TMP_DIR="$(mktemp -d /private/tmp/finderdockswitcher.XXXXXX)"
APP_DIR="$ROOT_DIR/dist/$PRODUCT_NAME.app"
MACOS_DIR="$APP_DIR/Contents/MacOS"
RESOURCES_DIR="$APP_DIR/Contents/Resources"
SOURCE_FILE="$ROOT_DIR/Sources/main.m"
ICON_SOURCE="$ROOT_DIR/scripts/generate_icon.m"
ICON_GENERATOR="$TMP_DIR/generate_icon"
ICON_PNG="$TMP_DIR/AppIcon.png"
ICON_BUNDLE_PNG="$RESOURCES_DIR/AppIcon.png"

cleanup() {
  rm -rf "$TMP_DIR"
}

trap cleanup EXIT

mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

clang \
  -framework Cocoa \
  -framework ApplicationServices \
  -fobjc-arc \
  -O2 \
  "$SOURCE_FILE" \
  -o "$MACOS_DIR/$PRODUCT_NAME"

clang \
  -framework AppKit \
  -fobjc-arc \
  -O2 \
  "$ICON_SOURCE" \
  -o "$ICON_GENERATOR"

"$ICON_GENERATOR" "$ICON_PNG"
cp "$ICON_PNG" "$ICON_BUNDLE_PNG"

cat > "$APP_DIR/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>FinderDockSwitcher</string>
    <key>CFBundleIdentifier</key>
    <string>local.codex.finder-dock-switcher</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon.png</string>
    <key>CFBundleName</key>
    <string>FinderDockSwitcher</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSAppleEventsUsageDescription</key>
    <string>FinderDockSwitcher needs Finder automation permission to list and focus open Finder windows.</string>
</dict>
</plist>
PLIST

echo "Built app bundle at:"
echo "$APP_DIR"
