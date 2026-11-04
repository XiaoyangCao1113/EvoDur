// SVG -> PDF + PNG with headless Chrome: node render_svg.js in.svg out_prefix
const puppeteer = require('puppeteer'); const fs = require('fs'); const path = require('path');
(async () => {
  const [inp, out] = process.argv.slice(2);
  const svg = fs.readFileSync(inp, 'utf8');
  const w = +svg.match(/width="([\d.]+)"/)[1], h = +svg.match(/height="([\d.]+)"/)[1];
  const b = await puppeteer.launch({headless: 'shell', args: ['--no-sandbox', '--allow-file-access-from-files']});
  const p = await b.newPage();
  await p.goto('file://' + path.resolve(inp));
  await p.evaluate(() => document.fonts.ready);
  await p.setViewport({width: w, height: h, deviceScaleFactor: 3});
  await p.screenshot({path: out + '.png'});
  await p.pdf({path: out + '.pdf', width: w + 'px', height: (h + 1) + 'px', printBackground: true, pageRanges: '1'});
  await b.close();
})();
