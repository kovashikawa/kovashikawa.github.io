// Capture each frame of the a/A video from scene.html.
// Usage: node capture.js [--fps 30] [--out DIR] [--scale 3] [--width 1280]
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

// parse args
function arg(name, def) {
  const i = process.argv.indexOf('--' + name);
  if (i > -1 && process.argv[i + 1]) return process.argv[i + 1];
  return def;
}
const FPS = parseInt(arg('fps', '30'), 10);
const WIDTH = parseInt(arg('width', '1280'), 10);
const SCALE = parseInt(arg('scale', '3'), 10);
const OUT = path.resolve(arg('out', './frames'));
const END = parseFloat(arg('end', '13.7'));
fs.mkdirSync(OUT, { recursive: true });

(async () => {
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-gpu',
      '--hide-scrollbars',
      '--force-color-profile=srgb',
      `--window-size=${WIDTH},${Math.round(WIDTH * 9 / 16)}`,
    ],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: WIDTH, height: Math.round(WIDTH * 9 / 16), deviceScaleFactor: SCALE });
  // freeze time: override requestAnimationFrame to do nothing; we drive frames manually.
  const url = 'file://' + path.resolve('./scene.html');
  await page.goto(url, { waitUntil: 'load' });

  // ensure fonts loaded
  await page.evaluate(() => document.fonts.ready.then(() => {}));
  await new Promise(r => setTimeout(r, 400));

  // Set FPS + end
  await page.evaluate((fps, end) => {
    window.VIDEO_FPS = fps;
    window.END_TIME = end;
    window.refreshToggleCenter();
  }, FPS, END);

  const totalFrames = Math.round(END * FPS);
  console.log('Rendering', totalFrames, 'frames at', FPS, 'fps,', WIDTH, 'x', Math.round(WIDTH*9/16), 'scale', SCALE);

  for (let f = 0; f < totalFrames; f++) {
    const fi = f;
    await page.evaluate((fi) => {
      window.frameIndex = fi;
      window.renderFrame();
    }, fi);
    // let the DOM settle (fonts/transforms) - small tick
    await new Promise(r => setTimeout(r, 5));
    const shot = await page.screenshot({ type: 'png' });
    const name = path.join(OUT, 'frame_' + String(fi).padStart(5, '0') + '.png');
    fs.writeFileSync(name, shot);
    if (fi % 90 === 0) console.log('  frame', fi, '/', totalFrames);
  }
  await browser.close();
  console.log('DONE', totalFrames, 'frames in', OUT);
})().catch(e => { console.error(e); process.exit(1); });
