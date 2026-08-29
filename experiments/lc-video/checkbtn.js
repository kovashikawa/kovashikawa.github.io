const puppeteer = require('puppeteer-core');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox','--disable-gpu','--hide-scrollbars'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 720 });
  await page.goto('file://' + require('path').resolve('./scene.html'), { waitUntil: 'load' });
  await page.evaluate(() => document.fonts.ready);
  await new Promise(r => setTimeout(r, 400));
  await page.evaluate((fps) => { window.VIDEO_FPS = fps; window.refreshToggleCenter(); }, 30);
  for (const t of [4.0, 5.3, 6.4, 8.3, 11.0]) {
    const r = await page.evaluate((t) => {
      window.frameIndex = Math.round(t*30); window.renderFrame();
      const a = document.querySelector('.lc-toggle .lc-a');
      const A = document.querySelector('.lc-toggle .lc-A');
      return {
        bodyLc: document.body.classList.contains('lc'),
        aWeight: getComputedStyle(a).fontWeight,
        AWeight: getComputedStyle(A).fontWeight,
      };
    }, t);
    console.log('t=' + t + 's  body.lc=' + r.bodyLc + '  a.weight=' + r.aWeight + '  A.weight=' + r.AWeight);
  }
  await browser.close();
})();
