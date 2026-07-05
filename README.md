# Bare™

Remove Instagram filters, restore natural colour, smooth skin, and upscale to 4K/8K.

This repository hosts the update feed for the Bare desktop app.

## How updates work

The app checks the `VERSION` file in this repo on startup. If it's newer than the
installed version, an update banner appears in the app with a one-click install.

## Releasing an update

1. Edit `photo_defilter.py` with your changes
2. Bump the `VERSION = "x.y.z"` constant at the top of the script
3. Update the `VERSION` file in this repo to match
4. Commit both — everyone gets the update banner on next launch
