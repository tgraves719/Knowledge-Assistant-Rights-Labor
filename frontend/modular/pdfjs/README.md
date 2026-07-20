Vendored PDF.js (pdfjs-dist 3.11.174, Apache-2.0) classic builds plus a
minimal same-origin viewer page. Used when the browser has no native inline
PDF renderer (`navigator.pdfViewerEnabled === false` — Android Chrome, iOS).
To upgrade: `npm install --no-save pdfjs-dist@<version>` and copy
`build/pdf.min.js` + `build/pdf.worker.min.js` here (classic scripts, not the
.mjs ESM builds — /static/modular serves .js with an explicit JS MIME type).
