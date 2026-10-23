/**
 * Đọc nhanh danh sách file từ NotebookLM sidebar
 * - Tái dùng chrome_profile đã đăng nhập
 * - Không cần hỏi AI, chỉ đọc DOM
 */
import { chromium } from "file:///D:/npm-global/node_modules/notebooklm-mcp/node_modules/patchright/index.mjs";
import { readFileSync, existsSync } from "fs";
import path from "path";

const NOTEBOOK_URL = "https://notebook.google.com/notebook/5936ccc7-5ed1-4764-9ccc-b5d8db377582";
const PROFILE_DIR   = "C:\\Users\\Thai anh ky\\AppData\\Local\\notebooklm-mcp\\Data\\chrome_profile";
const STATE_FILE    = "C:\\Users\\Thai anh ky\\AppData\\Local\\notebooklm-mcp\\Data\\browser_state\\state.json";

async function main() {
  console.log("🚀 Mở browser (headless)...");
  const browser = await chromium.launchPersistentContext(PROFILE_DIR, {
    headless: true,
    args: ["--no-sandbox", "--disable-blink-features=AutomationControlled"],
  });

  const page = await browser.newPage();

  // Load saved auth state nếu có
  if (existsSync(STATE_FILE)) {
    try {
      const state = JSON.parse(readFileSync(STATE_FILE, "utf8"));
      if (state.cookies) await browser.addCookies(state.cookies);
      console.log(`  ✅ Loaded ${state.cookies?.length ?? 0} cookies`);
    } catch (e) {
      console.warn("  ⚠️  Không load được state:", e.message);
    }
  }

  console.log("🌐 Điều hướng đến notebook...");
  await page.goto(NOTEBOOK_URL, { waitUntil: "domcontentloaded", timeout: 30000 });

  // Chờ sidebar nguồn xuất hiện
  console.log("⏳ Chờ sidebar sources...");
  try {
    await page.waitForSelector(".single-source-container", { timeout: 20000, state: "visible" });
  } catch {
    // Fallback: thử đọc header count
    console.log("  ⚠️  .single-source-container chưa thấy, thử đọc header...");
  }

  // Đọc số lượng từ header
  const headerText = await page.locator(".cover-subtitle-source-count").first().textContent({ timeout: 3000 }).catch(() => null);
  const totalMatch = headerText?.match(/(\d+)/);
  const totalFromHeader = totalMatch ? parseInt(totalMatch[1]) : "?";

  // Đọc tên từng source container
  const sourceNames = await page.evaluate(() => {
    const containers = document.querySelectorAll(".single-source-container");
    const names = [];
    containers.forEach((el) => {
      // Thử các class thông dụng cho title
      const titleEl =
        el.querySelector(".source-title") ||
        el.querySelector(".source-name") ||
        el.querySelector("[class*='title']") ||
        el.querySelector("[class*='name']") ||
        el.querySelector("span") ||
        el;
      const text = titleEl.textContent?.trim() || el.textContent?.trim() || "";
      // Lọc bỏ text quá dài hoặc trống
      if (text && text.length < 200) names.push(text);
    });
    return names;
  });

  // Lọc trùng lặp
  const unique = [...new Set(sourceNames.map(n => n.replace(/\s+/g, " ").trim()))].filter(Boolean);

  console.log("\n" + "=".repeat(60));
  console.log(`📚 Notebook: XLA - Gemini Notebook`);
  console.log(`📊 Số nguồn (header): ${totalFromHeader}`);
  console.log(`📋 Số nguồn đọc được từ DOM: ${unique.length}`);
  console.log("=".repeat(60));
  unique.forEach((name, i) => {
    console.log(`  ${i + 1}. ${name}`);
  });
  console.log("=".repeat(60));

  // Nếu không đọc được tên, dump raw HTML để debug
  if (unique.length === 0) {
    console.log("\n⚠️  Không đọc được source names, dump HTML để debug:");
    const html = await page.locator(".single-source-container").first().innerHTML({ timeout: 2000 }).catch(() => "N/A");
    console.log(html.slice(0, 1000));
  }

  await browser.close();
}

main().catch(e => {
  console.error("❌ Lỗi:", e.message);
  process.exit(1);
});
