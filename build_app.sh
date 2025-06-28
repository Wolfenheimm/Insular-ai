#!/bin/bash

# Build Insular.app bundle script
echo "🚀 Building Insular..."

# Build the release binary
cargo build --release -p insular

# Create app bundle structure
mkdir -p "Insular.app/Contents/MacOS" "Insular.app/Contents/Resources"

# Copy executable and icon
cp target/release/insular "Insular.app/Contents/MacOS/"
cp crates/zed/resources/insular-icon.png "Insular.app/Contents/Resources/"

# Create Info.plist
cat > "Insular.app/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleExecutable</key>
	<string>insular</string>
	<key>CFBundleIdentifier</key>
	<string>dev.insular.insular</string>
	<key>CFBundleName</key>
	<string>Insular</string>
	<key>CFBundleDisplayName</key>
	<string>Insular</string>
	<key>CFBundleVersion</key>
	<string>0.194.0</string>
	<key>CFBundleShortVersionString</key>
	<string>0.194.0</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleSignature</key>
	<string>????</string>
	<key>CFBundleIconFile</key>
	<string>insular-icon.png</string>
	<key>LSMinimumSystemVersion</key>
	<string>10.15</string>
	<key>NSHumanReadableCopyright</key>
	<string>Copyright © 2024 Insular AI. All rights reserved.</string>
	<key>NSHighResolutionCapable</key>
	<true/>
	<key>LSApplicationCategoryType</key>
	<string>public.app-category.developer-tools</string>
</dict>
</plist>
EOF

echo "✅ Insular.app created successfully!"
echo "🔍 Run 'open Insular.app' to launch" 