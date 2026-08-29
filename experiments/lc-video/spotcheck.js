// Spot-check key moments of the timeline as stills.
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

const WIDTH = 1280, SCALE = 3;
const SHOTS = [
  ['title_full', 1.5],
  ['page_default', 4.2],
  ['cursor_arrive', 4.7],
  ['flash_click', 5.0],
  ['lc_on', 5.6],
  ['settle_mid', 7.2],
  ['settle_done', 8.3],
  ['endcard', 11.0],
];

(async () => {
  const browser = await puppeteer.launch({
    executablePath: CHROME, headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--hide-scrollbars', '--force-color-profile=srgb', `--window-size=${WIDTH},${Math.round(WIDTH*9/16)}`],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: WIDTH, height: Math.round(WIDTH*9/16), deviceScaleFactor: SCALE });
  await page.goto('file://' + path.resolve('./scene.html'), { waitUntil: 'load' });
  await page.evaluate(() => document.fonts.ready);
  await new Promise(r => setTimeout(r, 400));
  await page.evaluate((fps) => { window.VIDEO_FPS = fps; window.refreshToggleCenter(); }, 30);

  for (const [name, t] of SHOTS) {
    await page.evaluate((t) => { window.frameIndex = Math.round(t * 30); window.renderFrame(); }, t);
    await new Promise(r => setTimeout(r, 20));
    const shot = await page.screenshot({ type: 'png' });
    fs.writeFileSync(path.join('./stills', name + '.png'), shot);
    console.log('shot', name, t, 's');
  }
  await browser.close();
  console.log('DONE');
})().catch(e => { console.error(e); process.exit(1); });
