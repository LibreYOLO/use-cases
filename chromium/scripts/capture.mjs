#!/usr/bin/env node
// Boot the static demo, point headed Chromium at it with a fake webcam,
// record 8 seconds, convert to GIF. Output: assets/hero.gif.

import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, renameSync, rmSync } from 'node:fs';
import { readdir } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { chromium } from 'playwright';
import ffmpegPath from 'ffmpeg-static';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..');
const ASSETS = join(ROOT, 'assets');
const SOURCE_IMG = join(ASSETS, 'demo-source.jpg');
const SAMPLE = join(ASSETS, '.capture-src.y4m');
const PORT = 5050;
const URL = `http://localhost:${PORT}/`;
const RECORD_MS = 8000;

mkdirSync(ASSETS, { recursive: true });

function generateSample() {
  if (existsSync(SAMPLE)) return;
  if (!existsSync(SOURCE_IMG)) throw new Error(`Missing ${SOURCE_IMG}. Bundled demo image is required.`);
  console.log('Generating fake-camera y4m from demo image...');
  // Loop the still image at 12 fps for 6 seconds with a slow ken-burns zoom so the GIF feels alive.
  // y4m output is uncompressed but small for a 6-second 480p clip, and Chromium's
  // --use-file-for-fake-video-capture loops it automatically.
  const filter = "scale=720:720:force_original_aspect_ratio=increase,crop=720:720,zoompan=z='min(zoom+0.0015,1.15)':d=72:s=720x720,fps=12";
  const args = ['-y', '-loop', '1', '-i', SOURCE_IMG, '-t', '6', '-vf', filter, '-pix_fmt', 'yuv420p', SAMPLE];
  const r = spawnSync(ffmpegPath, args, { stdio: 'inherit' });
  if (r.status !== 0) throw new Error('ffmpeg y4m generation failed');
}

function startServer() {
  console.log(`Starting static server on ${URL}`);
  const proc = spawn('npx', ['serve', '-l', String(PORT), ROOT], { stdio: 'ignore' });
  return proc;
}

async function waitForServer() {
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(URL);
      if (response.ok) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error('Server did not start within 10s');
}

async function record() {
  console.log('Launching chromium with fake webcam...');
  const browser = await chromium.launch({
    headless: true,
    args: [
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      `--use-file-for-fake-video-capture=${SAMPLE}`,
    ],
  });
  const context = await browser.newContext({
    viewport: { width: 720, height: 720 },
    recordVideo: { dir: ASSETS, size: { width: 720, height: 720 } },
    permissions: ['camera'],
  });
  const page = await context.newPage();
  page.on('console', (msg) => console.log(`[browser ${msg.type()}]`, msg.text()));
  page.on('pageerror', (err) => console.error('[browser error]', err.message));
  await page.goto(URL);
  console.log('Waiting for first detection...');
  await page.waitForFunction(() => window.__detectionsReady === true, { timeout: 60_000 });
  console.log(`Recording for ${RECORD_MS}ms...`);
  await page.waitForTimeout(RECORD_MS);
  const videoPath = await page.video().path();
  await context.close();
  await browser.close();
  return videoPath;
}

function toGif(webm, gif) {
  console.log('Converting to GIF...');
  // Two-pass palette for clean colors at small size. Trim to 6s, 320px wide,
  // 10fps, bayer dithering: balances visual quality vs ~2-3 MB target so
  // GitHub autoplays the GIF in the README.
  const palette = join(ASSETS, '.palette.png');
  const filtersPalette = 'fps=10,scale=320:-1:flags=lanczos,palettegen=stats_mode=diff';
  const filtersUse = '[0:v]fps=10,scale=320:-1:flags=lanczos[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=5';
  const r1 = spawnSync(ffmpegPath, ['-y', '-t', '6', '-i', webm, '-vf', filtersPalette, palette], { stdio: 'inherit' });
  if (r1.status !== 0) throw new Error('palettegen failed');
  const r2 = spawnSync(ffmpegPath, ['-y', '-t', '6', '-i', webm, '-i', palette, '-lavfi', filtersUse, gif], { stdio: 'inherit' });
  if (r2.status !== 0) throw new Error('paletteuse failed');
  rmSync(palette);
}

async function main() {
  generateSample();
  const server = startServer();
  try {
    await waitForServer();
    const webm = await record();
    const targetWebm = join(ASSETS, 'out.webm');
    renameSync(webm, targetWebm);
    const gif = join(ASSETS, 'hero.gif');
    toGif(targetWebm, gif);
    rmSync(targetWebm);
    // Clean up the playwright-created subdir if empty.
    try {
      const stragglers = await readdir(ASSETS);
      for (const s of stragglers) {
        if (s.endsWith('.webm')) rmSync(join(ASSETS, s));
      }
    } catch {}
    console.log(`\nDone: ${gif}`);
  } finally {
    server.kill('SIGTERM');
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
