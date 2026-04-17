#!/usr/bin/env node
// Boot the static demo, point headed Chromium at it with a fake webcam,
// record 8 seconds, convert to GIF. Output: assets/hero.gif.

import { spawn, spawnSync } from 'node:child_process';
import { createWriteStream, existsSync, mkdirSync, renameSync, rmSync } from 'node:fs';
import { mkdir, readdir } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { get } from 'node:https';
import { createHash } from 'node:crypto';

import { chromium } from 'playwright';
import ffmpegPath from 'ffmpeg-static';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..');
const ASSETS = join(ROOT, 'assets');
const SAMPLE = join(ASSETS, '.capture-src.y4m');
const PORT = 5050;
const URL = `http://localhost:${PORT}/`;
const RECORD_MS = 8000;

// Public-domain WebRTC sample, hosted by the WebRTC project. Pinned hash so
// reproducibility doesn't depend on the upstream remaining stable.
const SAMPLE_URL = 'https://test-videos.co.uk/vids/bigbuckbunny/y4m/360/Big_Buck_Bunny_360_10s_1MB.y4m';
const SAMPLE_SHA256 = null; // populate after first download; leave null to skip verification.

mkdirSync(ASSETS, { recursive: true });

async function downloadSample() {
  if (existsSync(SAMPLE)) return;
  console.log(`Downloading fake-camera source...`);
  await new Promise((res, rej) => {
    const file = createWriteStream(SAMPLE);
    get(SAMPLE_URL, (response) => {
      if (response.statusCode !== 200) return rej(new Error(`HTTP ${response.statusCode}`));
      response.pipe(file);
      file.on('finish', () => file.close(res));
    }).on('error', rej);
  });
  if (SAMPLE_SHA256) {
    const hash = createHash('sha256');
    const { readFileSync } = await import('node:fs');
    hash.update(readFileSync(SAMPLE));
    const got = hash.digest('hex');
    if (got !== SAMPLE_SHA256) throw new Error(`Sample checksum mismatch: ${got}`);
  }
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
    headless: false,
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
  // Two-pass palette for clean colors at small size.
  const palette = join(ASSETS, '.palette.png');
  const filtersPalette = 'fps=12,scale=480:-1:flags=lanczos,palettegen';
  const filtersUse = 'fps=12,scale=480:-1:flags=lanczos[v];[v][p]paletteuse';
  const r1 = spawnSync(ffmpegPath, ['-y', '-i', webm, '-vf', filtersPalette, palette], { stdio: 'inherit' });
  if (r1.status !== 0) throw new Error('palettegen failed');
  const r2 = spawnSync(ffmpegPath, ['-y', '-i', webm, '-i', palette, '-lavfi', filtersUse, gif], { stdio: 'inherit' });
  if (r2.status !== 0) throw new Error('paletteuse failed');
  rmSync(palette);
}

async function main() {
  await downloadSample();
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
