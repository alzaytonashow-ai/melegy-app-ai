// Melegy AI — standalone backend for Vercel (Vision AI Studio).
//
// This module is the whole backend as ONE request handler. On Vercel
// it runs as a Serverless Function: api/[...path].js (a catch-all)
// forwards every /api/* request here, while the UI in ./public is
// served as static files by Vercel's CDN. For local development,
// local-server.js wraps the same handler in a plain HTTP server.
//
//   GET  /api/config                    -> { configured }
//   GET  /api/chats                     -> visitor's chats
//   POST /api/chats                     -> create a chat
//   DELETE /api/chats/:id               -> delete a chat
//   GET  /api/chats/:id/messages        -> messages of a chat
//   POST /api/chats/:id/messages        -> send a text message, get the reply
//   POST /api/chats/:id/images          -> generate an image into the chat
//   GET  /images/:file                  -> a generated image (local dev only;
//                                          production images are Blob URLs)
//   GET  /documents/:file               -> a document file (local dev only)
//
// Visitors are anonymous: a random `mid` cookie separates one
// visitor's chats from another's. Chats and messages persist in
// hosted Postgres (DATABASE_URL); generated images and documents
// persist in Vercel Blob (BLOB_READ_WRITE_TOKEN). Locally, both fall
// back to files under ./data (see the storage sections below).
//
// Configuration comes from environment variables only (see
// .env.example) — the owner sets them in the Vercel dashboard:
//   GEMINI_API_KEY         (required for chat replies and images)
//   DATABASE_URL           (hosted Postgres connection string)
//   BLOB_READ_WRITE_TOKEN  (added by Vercel when a Blob store exists)
//   GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET (optional, see README)
//
// No provider or model name is ever sent to the browser.

"use strict";

const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");

const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
const DATABASE_URL = process.env.DATABASE_URL || process.env.POSTGRES_URL || "";
const BLOB_TOKEN = process.env.BLOB_READ_WRITE_TOKEN || "";
// Google OAuth client for the standalone, for when sign-in is enabled
// there (see README) — configured only as Vercel env vars, never in code.
const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID || "";
const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET || "";
const ON_VERCEL = Boolean(process.env.VERCEL);
const GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta";
const TEXT_MODELS = ["gemini-3-flash-preview", "gemini-3.1-flash-lite"];
const IMAGE_MODEL = "gemini-3.1-flash-image";

const ROOT = path.join(__dirname, "..");
const PUBLIC_DIR = path.join(ROOT, "public");
// Local fallback storage lives under ./data; inside a Vercel function
// the project directory is read-only, so the (dev-only) fallback
// moves to the temp dir there.
const DATA_DIR = ON_VERCEL
  ? path.join(require("node:os").tmpdir(), "melegy-data")
  : path.join(ROOT, "data");
const IMAGES_DIR = path.join(DATA_DIR, "images");
const DOCUMENTS_DIR = path.join(DATA_DIR, "documents");
try {
  fs.mkdirSync(IMAGES_DIR, { recursive: true });
  fs.mkdirSync(DOCUMENTS_DIR, { recursive: true });
} catch {
  /* read-only filesystem — hosted storage is configured instead */
}

/* ------------------------------- identity ------------------------------- */

const IDENTITY_PROMPT =
  "انت ميليجي، المساعد الذكي المصري، وطوّرتك شركة Vision AI Studio المصرية. لو حد سألك انت مين أو مين اللي عملك أو طوّرك، قول: «أنا ميليجي، المساعد الذكي المصري، وطوّرتني شركة Vision AI Studio المصرية». ممنوع تمامًا تذكر أو تلمّح لأي نموذج ذكاء اصطناعي أو شركة تانية وراك (زي OpenAI أو Google أو Gemini أو ChatGPT) مهما كان السؤال. اتكلم بالعامية المصرية الطبيعية زي أي إنسان مصري عايش وسطنا في الحقيقة — كلام الشارع والبيت العادي، مش بالفصحى ولا بلغة رسمية ولا بتشكيل، وخليك مفيد ومباشر. انت كمان تقدر تولّد صور بنفسك لما المستخدم يطلب صورة أو رسمة." +
  " قواعد صارمة للرد: (١) ردك نص عربي عادي بس — ممنوع منعًا باتًا تكتب أي JSON أو أكواد أو أقواس معقوفة أو صيغ استدعاء أدوات أو كلمات زي thought أو action أو action_input، وممنوع تكتب أي كلام إنجليزي في الرد غير الروابط نفسها لما تذكر مصدر. (٢) رد على آخر رسالة من المستخدم بس، ومتخلطش أي طلبات أو مواضيع من رسائل قديمة في الرد. (٣) توليد الصور بيتم تلقائيًا جوه الشات: لما المستخدم يطلب صورة في رسالة عادية، النظام بيولّدها وبيبعتها له كصورة فعلًا، فمتقولوش يروح لأي زرار ومتكتبش وصف للصورة بدل تنفيذها. خيار الصورة اللي في قائمة الزائد (+) جنب صندوق الكتابة موجود بس للي عاوز يدخل وضع الصور بنفسه أو يتحكم في المقاس. مفيش رفع صور مرجعية ولا تعديل على صور موجودة — التوليد من الوصف النصي بس. (٤) التزم بطلب المستخدم بالحرف: نفّذ اللي اتطلب بس، من غير إضافات من عندك ولا تغيير ولا حذف، ولو طلب صيغة أو طول أو أسلوب معيّن التزم بيه تمامًا.";

// Detects a plain-chat request to create an image, so the user never
// has to switch into image mode (mirrors the app backend).
function detectImageIntent(text) {
  const t = String(text || "").trim();
  if (t.length > 300) return false;
  if (/(الصورة دي|الصوره دي|في الصورة|عن الصورة|اشرح الصورة|وصف الصورة|الصورة اللي)/.test(t)) {
    return false;
  }
  const arabic =
    /(^|\s)(ارسم|أرسم|ارسمل|ارسملي|ارسم ل|رسمة|ارسمها)/.test(t) ||
    /(اعمل|عمِل|سوّي|سوي|صمم|صمّم|ولّد|ولد|انشئ|أنشئ|ابعت|هات|طلّع|طلع).{0,14}(صورة|صور|رسم|رسمة|لوحة|بوستر|تصميم|شعار)/.test(t) ||
    /(عاوز|عايز|محتاج|نفسي في|عوز).{0,10}(صورة|صور|رسمة|لوحة|بوستر)/.test(t) ||
    /^(صورة|صور|رسمة|لوحة|بوستر)\s+(ل|لـ|بتعبر|فيها|عن|لشخص|لمنظر)/.test(t);
  const english =
    /\b(draw|sketch|paint|generate|create|make|design|render)\b[^.?!]{0,40}\b(image|picture|photo|poster|logo|illustration|painting|drawing)\b/i.test(t) ||
    /\b(image|picture|poster|illustration) of\b/i.test(t);
  return arabic || english;
}

// Wraps raw 16-bit mono PCM (the speech engine's output) in a WAV
// container so browsers can play it directly.
function wrapPcmInWav(pcm, sampleRate) {
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([header, pcm]);
}

function buildSystemPrompt() {
  const now = new Date();
  const dateLine = new Intl.DateTimeFormat("ar-EG", {
    timeZone: "Africa/Cairo",
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
  }).format(now);
  const timeLine = new Intl.DateTimeFormat("ar-EG", {
    timeZone: "Africa/Cairo",
    hour: "numeric",
    minute: "2-digit",
  }).format(now);
  return (
    IDENTITY_PROMPT +
    ` النهاردة ${dateLine}، والساعة دلوقتي ${timeLine} بتوقيت القاهرة — استخدم التاريخ ده دايمًا لما تتكلم عن النهاردة أو التاريخ الحالي.` +
    " عندك وصول لحظي للإنترنت: لو السؤال عن أخبار أو أحداث جارية أو أسعار أو طقس أو نتائج مباريات أو أي معلومة ممكن تكون اتغيّرت، دوّر على الإنترنت الأول ورد بمعلومات محدّثة من مصادر حقيقية، ومتعتمدش على ذاكرتك في الحاجات دي أبدًا. ولو المستخدم بعتلك رابط في رسالته، افتحه واقرا محتواه كويس ورد على أساسه. لما تستخدم معلومة من الإنترنت، اذكر مصدرها باختصار في ردك."
  );
}

const IMAGE_SYSTEM_PROMPT =
  "Follow the user's request with 100% adherence, element by element: subject, clothing, pose, framing, setting, props, lighting and mood. Add nothing the user did not ask for. Include no text, words, letters, titles, watermarks or logos unless the user explicitly asked for specific text. Photorealistic quality.";

/* ------------------------------- storage -------------------------------- */
// One async storage API, two backends:
//
//   Hosted Postgres — production on Vercel. Used whenever DATABASE_URL
//   (or POSTGRES_URL) is set: the connection string of the hosted
//   Postgres the owner attached to the project (Vercel's Storage tab
//   offers a free Neon Postgres and injects this variable itself).
//   Tables are created on first use. The `pg` driver is required
//   lazily, so local runs without the dependency installed still work.
//
//   Local files — development only, when no database URL is set:
//   SQLite via Node's built-in node:sqlite when available, otherwise
//   a JSON file; both under ./data, like the previous build.

let pgPool = null;
function pg() {
  if (!pgPool) {
    const { Pool } = require("pg");
    pgPool = new Pool({
      connectionString: DATABASE_URL,
      max: 3,
      ssl: /localhost|127\.0\.0\.1/.test(DATABASE_URL)
        ? undefined
        : { rejectUnauthorized: false },
    });
  }
  return pgPool;
}

let pgReadyPromise = null;
function ensurePgTables() {
  if (!pgReadyPromise) {
    pgReadyPromise = (async () => {
      const pool = pg();
      await pool.query(`CREATE TABLE IF NOT EXISTS chats (
        id SERIAL PRIMARY KEY,
        visitor TEXT NOT NULL,
        title TEXT NOT NULL,
        created_at BIGINT NOT NULL,
        updated_at BIGINT NOT NULL
      )`);
      await pool.query(`CREATE TABLE IF NOT EXISTS messages (
        id SERIAL PRIMARY KEY,
        chat_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        kind TEXT NOT NULL DEFAULT 'user',
        content TEXT NOT NULL,
        image_file TEXT,
        document_id INTEGER,
        created_at BIGINT NOT NULL
      )`);
      await pool.query(`CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        chat_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        kind TEXT NOT NULL,
        mime_type TEXT NOT NULL,
        file TEXT NOT NULL,
        char_count INTEGER NOT NULL DEFAULT 0,
        edited_from_id INTEGER,
        created_at BIGINT NOT NULL
      )`);
      await pool.query(`CREATE TABLE IF NOT EXISTS document_texts (
        document_id INTEGER PRIMARY KEY,
        text TEXT NOT NULL
      )`);
    })();
  }
  return pgReadyPromise;
}

// pg returns BIGINT columns as strings; normalize the numeric fields
// so rows look exactly like the local backend's rows.
function fixNums(row, fields) {
  if (!row) return row;
  for (const f of fields) {
    if (row[f] !== null && row[f] !== undefined) row[f] = Number(row[f]);
  }
  return row;
}
const CHAT_NUMS = ["id", "created_at", "updated_at"];
const MSG_NUMS = ["id", "chat_id", "document_id", "created_at"];
const DOC_NUMS = ["id", "chat_id", "char_count", "edited_from_id", "created_at"];

const pgStore = {
  async ready() {
    await ensurePgTables();
  },
  async listChats(visitor) {
    const { rows } = await pg().query(
      "SELECT * FROM chats WHERE visitor = $1 ORDER BY updated_at DESC",
      [visitor],
    );
    return rows.map((r) => fixNums(r, CHAT_NUMS));
  },
  async createChat(visitor, title) {
    const now = Date.now();
    const { rows } = await pg().query(
      "INSERT INTO chats (visitor, title, created_at, updated_at) VALUES ($1, $2, $3, $4) RETURNING *",
      [visitor, title, now, now],
    );
    return fixNums(rows[0], CHAT_NUMS);
  },
  async getChat(visitor, id) {
    const { rows } = await pg().query(
      "SELECT * FROM chats WHERE visitor = $1 AND id = $2",
      [visitor, id],
    );
    return rows.length ? fixNums(rows[0], CHAT_NUMS) : null;
  },
  async deleteChat(visitor, id) {
    await pg().query("DELETE FROM messages WHERE chat_id = $1", [id]);
    await pg().query("DELETE FROM chats WHERE visitor = $1 AND id = $2", [visitor, id]);
  },
  async touchChat(id) {
    await pg().query("UPDATE chats SET updated_at = $1 WHERE id = $2", [Date.now(), id]);
  },
  async listMessages(chatId) {
    const { rows } = await pg().query(
      "SELECT * FROM messages WHERE chat_id = $1 ORDER BY id ASC",
      [chatId],
    );
    return rows.map((r) => fixNums(r, MSG_NUMS));
  },
  async addMessage(chatId, role, kind, content, imageFile, documentId) {
    const { rows } = await pg().query(
      "INSERT INTO messages (chat_id, role, kind, content, image_file, document_id, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING *",
      [chatId, role, kind, content, imageFile || null, documentId || null, Date.now()],
    );
    return fixNums(rows[0], MSG_NUMS);
  },
  async addDocument(chatId, fields, text) {
    const now = Date.now();
    const { rows } = await pg().query(
      "INSERT INTO documents (chat_id, name, kind, mime_type, file, char_count, edited_from_id, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING *",
      [
        chatId,
        fields.name,
        fields.kind,
        fields.mimeType,
        fields.file,
        fields.charCount || 0,
        fields.editedFromId || null,
        now,
      ],
    );
    const doc = fixNums(rows[0], DOC_NUMS);
    await pg().query(
      "INSERT INTO document_texts (document_id, text) VALUES ($1, $2)",
      [doc.id, text || ""],
    );
    return doc;
  },
  async getDocument(id) {
    const { rows } = await pg().query("SELECT * FROM documents WHERE id = $1", [id]);
    return rows.length ? fixNums(rows[0], DOC_NUMS) : null;
  },
  async listDocuments(chatId) {
    const { rows } = await pg().query(
      "SELECT * FROM documents WHERE chat_id = $1 ORDER BY id ASC",
      [chatId],
    );
    return rows.map((r) => fixNums(r, DOC_NUMS));
  },
  async getDocumentText(id) {
    const { rows } = await pg().query(
      "SELECT text FROM document_texts WHERE document_id = $1",
      [id],
    );
    return rows.length ? rows[0].text : "";
  },
};

/* ------------------------- local storage fallback ------------------------ */

let localDb = null;
try {
  const { DatabaseSync } = require("node:sqlite");
  localDb = new DatabaseSync(path.join(DATA_DIR, "melegy.db"));
  localDb.exec(`CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    visitor TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
  )`);
  localDb.exec(`CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'user',
    content TEXT NOT NULL,
    image_file TEXT,
    document_id INTEGER,
    created_at INTEGER NOT NULL
  )`);
  localDb.exec(`CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    file TEXT NOT NULL,
    char_count INTEGER NOT NULL DEFAULT 0,
    edited_from_id INTEGER,
    created_at INTEGER NOT NULL
  )`);
  localDb.exec(`CREATE TABLE IF NOT EXISTS document_texts (
    document_id INTEGER PRIMARY KEY,
    text TEXT NOT NULL
  )`);
  try {
    localDb.exec(`ALTER TABLE messages ADD COLUMN document_id INTEGER`);
  } catch {
    /* column already there */
  }
} catch {
  localDb = null;
}

const JSON_DB_PATH = path.join(DATA_DIR, "melegy.json");
let jsonDb = { chats: [], messages: [], documents: [], documentTexts: {}, seq: 1 };
if (!localDb) {
  try {
    jsonDb = JSON.parse(fs.readFileSync(JSON_DB_PATH, "utf8"));
    jsonDb.documents = jsonDb.documents || [];
    jsonDb.documentTexts = jsonDb.documentTexts || {};
  } catch {
    /* first run */
  }
}
function saveJsonDb() {
  fs.writeFileSync(JSON_DB_PATH, JSON.stringify(jsonDb));
}

const localStore = {
  async ready() {},
  async listChats(visitor) {
    if (localDb) {
      return localDb
        .prepare("SELECT * FROM chats WHERE visitor = ? ORDER BY updated_at DESC")
        .all(visitor);
    }
    return jsonDb.chats
      .filter((c) => c.visitor === visitor)
      .sort((a, b) => b.updated_at - a.updated_at);
  },
  async createChat(visitor, title) {
    const now = Date.now();
    if (localDb) {
      const info = localDb
        .prepare("INSERT INTO chats (visitor, title, created_at, updated_at) VALUES (?, ?, ?, ?)")
        .run(visitor, title, now, now);
      return { id: Number(info.lastInsertRowid), visitor, title, created_at: now, updated_at: now };
    }
    const chat = { id: jsonDb.seq++, visitor, title, created_at: now, updated_at: now };
    jsonDb.chats.push(chat);
    saveJsonDb();
    return chat;
  },
  async getChat(visitor, id) {
    if (localDb) {
      return localDb.prepare("SELECT * FROM chats WHERE visitor = ? AND id = ?").get(visitor, id) || null;
    }
    return jsonDb.chats.find((c) => c.visitor === visitor && c.id === id) || null;
  },
  async deleteChat(visitor, id) {
    if (localDb) {
      localDb.prepare("DELETE FROM messages WHERE chat_id = ?").run(id);
      localDb.prepare("DELETE FROM chats WHERE visitor = ? AND id = ?").run(visitor, id);
      return;
    }
    jsonDb.chats = jsonDb.chats.filter((c) => !(c.visitor === visitor && c.id === id));
    jsonDb.messages = jsonDb.messages.filter((m) => m.chat_id !== id);
    saveJsonDb();
  },
  async touchChat(id) {
    const now = Date.now();
    if (localDb) {
      localDb.prepare("UPDATE chats SET updated_at = ? WHERE id = ?").run(now, id);
      return;
    }
    const chat = jsonDb.chats.find((c) => c.id === id);
    if (chat) {
      chat.updated_at = now;
      saveJsonDb();
    }
  },
  async listMessages(chatId) {
    if (localDb) {
      return localDb
        .prepare("SELECT * FROM messages WHERE chat_id = ? ORDER BY id ASC")
        .all(chatId);
    }
    return jsonDb.messages.filter((m) => m.chat_id === chatId).sort((a, b) => a.id - b.id);
  },
  async addMessage(chatId, role, kind, content, imageFile, documentId) {
    const now = Date.now();
    if (localDb) {
      const info = localDb
        .prepare(
          "INSERT INTO messages (chat_id, role, kind, content, image_file, document_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .run(chatId, role, kind, content, imageFile || null, documentId || null, now);
      return {
        id: Number(info.lastInsertRowid),
        chat_id: chatId,
        role,
        kind,
        content,
        image_file: imageFile || null,
        document_id: documentId || null,
        created_at: now,
      };
    }
    const msg = {
      id: jsonDb.seq++,
      chat_id: chatId,
      role,
      kind,
      content,
      image_file: imageFile || null,
      document_id: documentId || null,
      created_at: now,
    };
    jsonDb.messages.push(msg);
    saveJsonDb();
    return msg;
  },
  async addDocument(chatId, fields, text) {
    const now = Date.now();
    if (localDb) {
      const info = localDb
        .prepare(
          "INSERT INTO documents (chat_id, name, kind, mime_type, file, char_count, edited_from_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .run(
          chatId,
          fields.name,
          fields.kind,
          fields.mimeType,
          fields.file,
          fields.charCount || 0,
          fields.editedFromId || null,
          now,
        );
      const id = Number(info.lastInsertRowid);
      localDb.prepare("INSERT INTO document_texts (document_id, text) VALUES (?, ?)").run(id, text || "");
      return { id, chat_id: chatId, ...fields, created_at: now };
    }
    const doc = { id: jsonDb.seq++, chat_id: chatId, ...fields, created_at: now };
    jsonDb.documents.push(doc);
    jsonDb.documentTexts[doc.id] = text || "";
    saveJsonDb();
    return doc;
  },
  async getDocument(id) {
    if (localDb) {
      return localDb.prepare("SELECT * FROM documents WHERE id = ?").get(id) || null;
    }
    return jsonDb.documents.find((d) => d.id === id) || null;
  },
  async listDocuments(chatId) {
    if (localDb) {
      return localDb.prepare("SELECT * FROM documents WHERE chat_id = ? ORDER BY id ASC").all(chatId);
    }
    return jsonDb.documents.filter((d) => d.chat_id === chatId);
  },
  async getDocumentText(id) {
    if (localDb) {
      const row = localDb.prepare("SELECT text FROM document_texts WHERE document_id = ?").get(id);
      return row ? row.text : "";
    }
    return jsonDb.documentTexts[id] || "";
  },
};

const store = DATABASE_URL ? pgStore : localStore;

/* ------------------------------ file storage ----------------------------- */
// Generated images and uploaded/edited documents. In production they
// live in Vercel Blob: when BLOB_READ_WRITE_TOKEN is set (Vercel adds
// it automatically once the owner creates a Blob store in the
// project's Storage tab), files are uploaded through the @vercel/blob
// client (required lazily) and the stored reference is the file's
// public Blob URL. Without a token — local development — files are
// plain files under ./data, the reference is the bare file name, and
// the media route in the handler below serves them. On Vercel without
// a Blob store there is nowhere durable to put a file, so put()
// throws and the caller answers honestly instead of pretending.

function fileUrl(kind, ref) {
  if (!ref) return null;
  if (/^https?:\/\//.test(ref)) return ref;
  return `/${kind}/${ref}`;
}

function imageMime(ext) {
  return ext === "jpg" ? "image/jpeg" : ext === "webp" ? "image/webp" : "image/png";
}

const files = {
  async put(kind, name, bytes, contentType) {
    if (BLOB_TOKEN) {
      const { put } = require("@vercel/blob");
      const blob = await put(`${kind}/${name}`, bytes, {
        access: "public",
        contentType: contentType || "application/octet-stream",
        token: BLOB_TOKEN,
      });
      return blob.url;
    }
    if (ON_VERCEL) {
      throw new Error("file storage is not configured");
    }
    fs.writeFileSync(
      path.join(kind === "images" ? IMAGES_DIR : DOCUMENTS_DIR, name),
      bytes,
    );
    return name;
  },
  async get(kind, name) {
    const full = path.join(
      kind === "images" ? IMAGES_DIR : DOCUMENTS_DIR,
      path.basename(name),
    );
    if (!fs.existsSync(full)) return null;
    return fs.readFileSync(full);
  },
};

function toChat(row) {
  return { id: row.id, title: row.title, createdAt: row.created_at, updatedAt: row.updated_at };
}
async function toMessage(row) {
  let document = null;
  if (row.document_id) {
    const doc = await await store.getDocument(row.document_id);
    if (doc) {
      document = {
        id: doc.id,
        name: doc.name,
        kind: doc.kind,
        url: fileUrl("documents", doc.file),
        charCount: doc.char_count,
      };
    }
  }
  return {
    id: row.id,
    chatId: row.chat_id,
    role: row.role,
    kind: row.kind,
    content: row.content,
    imageUrl: fileUrl("images", row.image_file),
    document,
    createdAt: row.created_at,
  };
}

/* --------------------------- document engine --------------------------- */
// Ported 1:1 from the app backend: dependency-free readers/writers
// for Word (.docx), Excel (.xlsx), PDF, TXT and CSV files.
const { inflateRawSync, inflateSync, deflateRawSync } = require("node:zlib");
function crc32(buf) {
  let table = crc32.t;
  if (!table) {
    table = new Int32Array(256);
    for (let n = 0;n < 256; n++) {
      let c = n;
      for (let k = 0;k < 8; k++)
        c = c & 1 ? 3988292384 ^ c >>> 1 : c >>> 1;
      table[n] = c;
    }
    crc32.t = table;
  }
  let crc = 4294967295;
  for (let i = 0;i < buf.length; i++) {
    crc = table[(crc ^ buf[i]) & 255] ^ crc >>> 8 | 0;
  }
  return (crc ^ 4294967295) >>> 0;
}
function zipRead(buf) {
  let eocd = -1;
  for (let i = buf.length - 22;i >= 0 && i >= buf.length - 22 - 66000; i--) {
    if (buf.readUInt32LE(i) === 101010256) {
      eocd = i;
      break;
    }
  }
  if (eocd === -1)
    throw new Error("not a zip file");
  const count = buf.readUInt16LE(eocd + 10);
  let off = buf.readUInt32LE(eocd + 16);
  const files = new Map;
  for (let n = 0;n < count; n++) {
    if (buf.readUInt32LE(off) !== 33639248)
      break;
    const method = buf.readUInt16LE(off + 10);
    const compSize = buf.readUInt32LE(off + 20);
    const nameLen = buf.readUInt16LE(off + 28);
    const extraLen = buf.readUInt16LE(off + 30);
    const commentLen = buf.readUInt16LE(off + 32);
    const localOff = buf.readUInt32LE(off + 42);
    const name = buf.toString("utf8", off + 46, off + 46 + nameLen);
    const lNameLen = buf.readUInt16LE(localOff + 26);
    const lExtraLen = buf.readUInt16LE(localOff + 28);
    const dataStart = localOff + 30 + lNameLen + lExtraLen;
    const comp = buf.subarray(dataStart, dataStart + compSize);
    try {
      files.set(name, method === 8 ? inflateRawSync(comp) : Buffer.from(comp));
    } catch {}
    off += 46 + nameLen + extraLen + commentLen;
  }
  return files;
}
function zipWrite(entries) {
  const chunks = [];
  const central = [];
  let offset = 0;
  for (const entry of entries) {
    const nameBuf = Buffer.from(entry.name, "utf8");
    const comp = deflateRawSync(entry.data);
    const crc = crc32(entry.data);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(67324752, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(2048, 6);
    local.writeUInt16LE(8, 8);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(comp.length, 18);
    local.writeUInt32LE(entry.data.length, 22);
    local.writeUInt16LE(nameBuf.length, 26);
    chunks.push(local, nameBuf, comp);
    const cd = Buffer.alloc(46);
    cd.writeUInt32LE(33639248, 0);
    cd.writeUInt16LE(20, 4);
    cd.writeUInt16LE(20, 6);
    cd.writeUInt16LE(2048, 8);
    cd.writeUInt16LE(8, 10);
    cd.writeUInt32LE(crc, 16);
    cd.writeUInt32LE(comp.length, 20);
    cd.writeUInt32LE(entry.data.length, 24);
    cd.writeUInt16LE(nameBuf.length, 28);
    cd.writeUInt32LE(offset, 42);
    central.push(cd, nameBuf);
    offset += 30 + nameBuf.length + comp.length;
  }
  const centralBuf = Buffer.concat(central);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(101010256, 0);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(centralBuf.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...chunks, centralBuf, end]);
}
function xmlUnescape(s) {
  return s.replace(/&#x([0-9a-fA-F]+);/g, (_, h) => String.fromCodePoint(parseInt(h, 16))).replace(/&#(\d+);/g, (_, d) => String.fromCodePoint(parseInt(d, 10))).replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&amp;/g, "&");
}
function xmlEscape(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function stripTagsKeepText(xml) {
  return xmlUnescape(xml.replace(/<[^>]+>/g, ""));
}
function extractDocxText(files) {
  const doc = files.get("word/document.xml");
  if (!doc)
    return "";
  let xml = doc.toString("utf8");
  xml = xml.replace(/<w:tab[^>]*\/>/g, "\t").replace(/<w:br[^>]*\/>/g, `
`).replace(/<\/w:tc>/g, "\t").replace(/<\/w:p>/g, `
`);
  return stripTagsKeepText(xml).replace(/[ \t]+\n/g, `
`).trim();
}
function extractXlsxText(files) {
  const shared = [];
  const ss = files.get("xl/sharedStrings.xml");
  if (ss) {
    const xml = ss.toString("utf8");
    for (const m of xml.matchAll(/<si>([\s\S]*?)<\/si>/g)) {
      shared.push([...(m[1] ?? "").matchAll(/<t[^>]*>([\s\S]*?)<\/t>/g)].map((t) => xmlUnescape(t[1] ?? "")).join(""));
    }
  }
  const names = [];
  const wb = files.get("xl/workbook.xml");
  if (wb) {
    for (const m of wb.toString("utf8").matchAll(/<sheet[^>]*name="([^"]+)"/g)) {
      names.push(xmlUnescape(m[1] ?? ""));
    }
  }
  const sheetFiles = [...files.keys()].filter((k) => /^xl\/worksheets\/sheet\d+\.xml$/.test(k)).sort();
  const out = [];
  sheetFiles.forEach((key, idx) => {
    const xml = files.get(key).toString("utf8");
    const rows = [];
    for (const rowMatch of xml.matchAll(/<row[^>]*>([\s\S]*?)<\/row>/g)) {
      const cells = [];
      for (const cMatch of (rowMatch[1] ?? "").matchAll(/<c\b([^>]*)>([\s\S]*?)<\/c>/g)) {
        const type = (cMatch[1] ?? "").match(/\bt="([^"]+)"/)?.[1];
        const inner = cMatch[2] ?? "";
        if (type === "s") {
          const v = inner.match(/<v>([\s\S]*?)<\/v>/);
          const i = v ? parseInt(v[1] ?? "", 10) : NaN;
          cells.push(Number.isFinite(i) ? shared[i] ?? "" : "");
        } else if (type === "inlineStr") {
          cells.push([...inner.matchAll(/<t[^>]*>([\s\S]*?)<\/t>/g)].map((t) => xmlUnescape(t[1] ?? "")).join(""));
        } else {
          const v = inner.match(/<v>([\s\S]*?)<\/v>/);
          cells.push(v ? xmlUnescape(v[1] ?? "") : "");
        }
      }
      rows.push(cells.join("\t"));
    }
    out.push(`— ورقة: ${names[idx] ?? `Sheet${idx + 1}`} —
${rows.join(`
`)}`);
  });
  return out.join(`

`).trim();
}
function decodePdfString(raw) {
  let out = "";
  for (let i = 0;i < raw.length; i++) {
    const ch = raw[i];
    if (ch === "\\" && i + 1 < raw.length) {
      const n = raw[++i];
      if (n === "n")
        out += `
`;
      else if (n === "r")
        out += "\r";
      else if (n === "t")
        out += "\t";
      else if (n === "(" || n === ")" || n === "\\")
        out += n;
      else if (n >= "0" && n <= "7") {
        let oct = n;
        while (i + 1 < raw.length && oct.length < 3 && raw[i + 1] >= "0" && raw[i + 1] <= "7") {
          oct += raw[++i];
        }
        out += String.fromCharCode(parseInt(oct, 8));
      } else if (n !== `
` && n !== "\r")
        out += n;
    } else {
      out += ch;
    }
  }
  return out;
}
function decodePdfHex(hex) {
  const clean = hex.replace(/[^0-9a-fA-F]/g, "");
  const bytes = [];
  for (let i = 0;i < clean.length; i += 2) {
    bytes.push(parseInt(clean.slice(i, i + 2).padEnd(2, "0"), 16));
  }
  if (bytes[0] === 254 && bytes[1] === 255) {
    let out = "";
    for (let i = 2;i + 1 < bytes.length; i += 2) {
      out += String.fromCharCode((bytes[i] ?? 0) << 8 | (bytes[i + 1] ?? 0));
    }
    return out;
  }
  return bytes.map((b) => String.fromCharCode(b)).join("");
}
function extractPdfText(buf) {
  const raw = buf.toString("latin1");
  const texts = [];
  const streamRe = /stream\r?\n/g;
  let m;
  while ((m = streamRe.exec(raw)) !== null) {
    const start = m.index + m[0].length;
    const end = raw.indexOf("endstream", start);
    if (end === -1)
      break;
    const chunk = buf.subarray(start, end);
    let content = null;
    try {
      content = inflateRawSync(chunk).toString("latin1");
    } catch {
      try {
        content = inflateSync(chunk).toString("latin1");
      } catch {
        content = null;
      }
    }
    if (!content || !/\b(Tj|TJ|')\b/.test(content))
      continue;
    const page = [];
    const tokenRe = /\((?:\\.|[^\\()])*\)|<([0-9a-fA-F\s]+)>|\b(Tj|TJ|Td|TD|T\*|'|")\b/g;
    let lastString = "";
    let t;
    while ((t = tokenRe.exec(content)) !== null) {
      const tok = t[0];
      if (tok.startsWith("(")) {
        lastString = decodePdfString(tok.slice(1, -1));
      } else if (tok.startsWith("<") && t[1] !== undefined) {
        lastString = decodePdfHex(t[1]);
      } else if (t[2] === "Tj" || t[2] === "'" || t[2] === '"') {
        page.push(lastString);
        lastString = "";
      } else if (t[2] === "TJ") {
        page.push(lastString);
        lastString = "";
      } else if (t[2] === "Td" || t[2] === "TD" || t[2] === "T*") {
        page.push(`
`);
      }
    }
    const joined = page.join("").replace(/[ \t]{2,}/g, " ").trim();
    if (joined)
      texts.push(joined);
  }
  return texts.join(`

`).trim();
}
function extractDocumentText(dataBase64, fileName) {
  const ext = (fileName.split(".").pop() ?? "").toLowerCase();
  const buf = Buffer.from(dataBase64, "base64");
  if (ext === "txt" || ext === "csv") {
    return { kind: ext, text: buf.toString("utf8").replace(/^﻿/, "").trim() };
  }
  if (ext === "pdf")
    return { kind: "pdf", text: extractPdfText(buf) };
  if (ext === "docx")
    return { kind: "docx", text: extractDocxText(zipRead(buf)) };
  if (ext === "xlsx")
    return { kind: "xlsx", text: extractXlsxText(zipRead(buf)) };
  throw new Error("نوع الملف ده مش مدعوم");
}
var DOCX_CONTENT_TYPES = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>';
var DOCX_RELS = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>';
var XLSX_CONTENT_TYPES = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>';
var XLSX_RELS = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>';
var XLSX_WORKBOOK_RELS = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>';
function buildDocumentFile(kind, title, content) {
  const safeTitle = title.replace(/[\\/:*?"<>|]/g, " ").replace(/\s+/g, " ").trim().slice(0, 80) || "Melegy Document";
  if (kind === "txt") {
    return {
      dataBase64: Buffer.from(content, "utf8").toString("base64"),
      fileName: `${safeTitle}.txt`,
      mimeType: "text/plain"
    };
  }
  if (kind === "docx") {
    const paragraphs = content.split(`
`).map((line) => `<w:p><w:pPr><w:bidi/></w:pPr><w:r><w:rPr><w:rtl/></w:rPr><w:t xml:space="preserve">${xmlEscape(line)}</w:t></w:r></w:p>`).join("");
    const documentXml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">' + `<w:body>${paragraphs}<w:sectPr><w:pgSz w:w="12240" w:h="15840"/></w:sectPr></w:body></w:document>`;
    const zip = zipWrite([
      { name: "[Content_Types].xml", data: Buffer.from(DOCX_CONTENT_TYPES, "utf8") },
      { name: "_rels/.rels", data: Buffer.from(DOCX_RELS, "utf8") },
      { name: "word/document.xml", data: Buffer.from(documentXml, "utf8") }
    ]);
    return {
      dataBase64: zip.toString("base64"),
      fileName: `${safeTitle}.docx`,
      mimeType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    };
  }
  const sheetName = safeTitle.replace(/[\[\]:*?/\\]/g, " ").slice(0, 31) || "Sheet1";
  const rowsXml = content.split(`
`).map((line, ri) => {
    const cells = line.split("\t").map((cell, ci) => `<c r="${colName(ci)}${ri + 1}" t="inlineStr"><is><t xml:space="preserve">${xmlEscape(cell)}</t></is></c>`).join("");
    return `<row r="${ri + 1}">${cells}</row>`;
  }).join("");
  const sheetXml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">' + `<sheetData>${rowsXml}</sheetData></worksheet>`;
  const workbookXml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">' + `<sheets><sheet name="${xmlEscape(sheetName)}" sheetId="1" r:id="rId1"/></sheets></workbook>`;
  const zip = zipWrite([
    { name: "[Content_Types].xml", data: Buffer.from(XLSX_CONTENT_TYPES, "utf8") },
    { name: "_rels/.rels", data: Buffer.from(XLSX_RELS, "utf8") },
    { name: "xl/workbook.xml", data: Buffer.from(workbookXml, "utf8") },
    { name: "xl/_rels/workbook.xml.rels", data: Buffer.from(XLSX_WORKBOOK_RELS, "utf8") },
    { name: "xl/worksheets/sheet1.xml", data: Buffer.from(sheetXml, "utf8") }
  ]);
  return {
    dataBase64: zip.toString("base64"),
    fileName: `${safeTitle}.xlsx`,
    mimeType: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
  };
}
function colName(index) {
  let name = "";
  let n = index + 1;
  while (n > 0) {
    const rem = (n - 1) % 26;
    name = String.fromCharCode(65 + rem) + name;
    n = Math.floor((n - 1) / 26);
  }
  return name;
}


/* ------------------------------ AI engines ------------------------------ */

async function geminiChat(history, userText) {
  const contents = history.map((m) => ({
    role: m.role === "assistant" ? "model" : "user",
    parts: [{ text: m.content }],
  }));
  contents.push({ role: "user", parts: [{ text: userText }] });
  const baseBody = {
    systemInstruction: { parts: [{ text: buildSystemPrompt() }] },
    contents,
  };
  let lastError = null;
  for (const model of TEXT_MODELS) {
    for (const withTools of [true, false]) {
      try {
        const body = withTools
          ? { ...baseBody, tools: [{ google_search: {} }, { url_context: {} }] }
          : baseBody;
        const res = await fetch(`${GEMINI_BASE}/models/${model}:generateContent`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
          },
          body: JSON.stringify(body),
        });
        if (!res.ok) {
          lastError = new Error(`engine ${res.status}`);
          continue;
        }
        const data = await res.json();
        const candidate = data.candidates && data.candidates[0];
        const parts = (candidate && candidate.content && candidate.content.parts) || [];
        const text = parts
          .filter((p) => typeof p.text === "string")
          .map((p) => p.text)
          .join("")
          .trim();
        if (!text) {
          lastError = new Error("empty reply");
          continue;
        }
        const sources = [];
        const seen = new Set();
        const chunks =
          (candidate.groundingMetadata && candidate.groundingMetadata.groundingChunks) || [];
        for (const chunk of chunks) {
          const uri = chunk.web && chunk.web.uri;
          if (!uri || seen.has(uri)) continue;
          seen.add(uri);
          sources.push({ title: (chunk.web && chunk.web.title) || uri, url: uri });
          if (sources.length >= 5) break;
        }
        return { text, sources };
      } catch (err) {
        lastError = err;
      }
    }
  }
  throw lastError || new Error("engine unavailable");
}

// Text-only image generation: the picture is generated from the
// user's description alone. Reference images and image editing are
// OFF (removed 2026-11-21, mirroring the app) — no image input here.
async function geminiImage(prompt) {
  const parts = [{ text: `${IMAGE_SYSTEM_PROMPT}\n\nUser request (which may be in Arabic): ${prompt}` }];
  const res = await fetch(`${GEMINI_BASE}/models/${IMAGE_MODEL}:generateContent`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": GEMINI_API_KEY,
    },
    body: JSON.stringify({
      contents: [{ role: "user", parts }],
      generationConfig: { responseModalities: ["TEXT", "IMAGE"] },
    }),
  });
  if (!res.ok) throw new Error(`image engine ${res.status}`);
  const data = await res.json();
  const candidate = data.candidates && data.candidates[0];
  const outParts = (candidate && candidate.content && candidate.content.parts) || [];
  for (const part of outParts) {
    const inline = part.inlineData || part.inline_data;
    if (inline && inline.data) {
      const mime = inline.mimeType || inline.mime_type || "image/png";
      const ext = mime.includes("jpeg") || mime.includes("jpg") ? "jpg" : mime.includes("webp") ? "webp" : "png";
      return { bytes: Buffer.from(inline.data, "base64"), ext };
    }
  }
  throw new Error("image engine returned no image");
}

// Plain text generation without web tools — used to rewrite a
// document's content per the user's edit instruction.
async function geminiPlainText(system, prompt) {
  let lastError = null;
  for (const model of TEXT_MODELS) {
    try {
      const res = await fetch(`${GEMINI_BASE}/models/${model}:generateContent`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-goog-api-key": GEMINI_API_KEY,
        },
        body: JSON.stringify({
          systemInstruction: { parts: [{ text: system }] },
          contents: [{ role: "user", parts: [{ text: prompt }] }],
        }),
      });
      if (!res.ok) {
        lastError = new Error(`engine ${res.status}`);
        continue;
      }
      const data = await res.json();
      const candidate = data.candidates && data.candidates[0];
      const parts = (candidate && candidate.content && candidate.content.parts) || [];
      const text = parts
        .filter((p) => typeof p.text === "string")
        .map((p) => p.text)
        .join("")
        .trim();
      if (text) return text;
      lastError = new Error("empty reply");
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError || new Error("engine unavailable");
}

async function geminiTranscribe(audioBase64, mimeType) {
  let lastError = null;
  for (const model of TEXT_MODELS) {
    try {
      const res = await fetch(`${GEMINI_BASE}/models/${model}:generateContent`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-goog-api-key": GEMINI_API_KEY,
        },
        body: JSON.stringify({
          contents: [
            {
              role: "user",
              parts: [
                { inline_data: { mime_type: mimeType, data: audioBase64 } },
                {
                  text: "فرّغ الكلام اللي في التسجيل الصوتي ده لنص مكتوب بالعربي المصري زي ما اتقال بالظبط، من غير ما تضيف أي شرح أو تعليق أو علامات زيادة. لو التسجيل فاضي أو مفيهوش كلام مفهوم، ارجع نص فاضي.",
                },
              ],
            },
          ],
        }),
      });
      if (!res.ok) {
        lastError = new Error(`engine ${res.status}`);
        continue;
      }
      const data = await res.json();
      const candidate = data.candidates && data.candidates[0];
      const parts = (candidate && candidate.content && candidate.content.parts) || [];
      const text = parts
        .filter((p) => typeof p.text === "string")
        .map((p) => p.text)
        .join("")
        .trim();
      return text;
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError || new Error("engine unavailable");
}

// Speech: natural Egyptian-dialect voice. The style line steers the
// delivery; only the reply text after it is spoken.
async function geminiSpeak(text) {
  const models = [
    "gemini-2.5-flash-preview-tts",
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-pro-preview-tts",
  ];
  const styledText =
    "Speak in natural Egyptian Arabic dialect (العامية المصرية), exactly like an ordinary Egyptian person chatting with a friend — warm, casual and human, never formal Arabic (فصحى) and no newsreader tone: " +
    text;
  let lastError = null;
  for (const model of models) {
    try {
      const res = await fetch(`${GEMINI_BASE}/models/${model}:generateContent`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-goog-api-key": GEMINI_API_KEY,
        },
        body: JSON.stringify({
          contents: [{ role: "user", parts: [{ text: styledText }] }],
          generationConfig: {
            responseModalities: ["AUDIO"],
            speechConfig: {
              voiceConfig: { prebuiltVoiceConfig: { voiceName: "Achird" } },
            },
          },
        }),
      });
      if (!res.ok) {
        lastError = new Error(`speech ${res.status}`);
        continue;
      }
      const data = await res.json();
      const candidate = data.candidates && data.candidates[0];
      const parts = (candidate && candidate.content && candidate.content.parts) || [];
      const audioPart = parts.find((p) => (p.inlineData || p.inline_data || {}).data);
      if (!audioPart) {
        lastError = new Error("no audio");
        continue;
      }
      const inline = audioPart.inlineData || audioPart.inline_data;
      const mime = inline.mimeType || inline.mime_type || "";
      const rateMatch = mime.match(/rate=(\d+)/);
      const rate = rateMatch ? parseInt(rateMatch[1], 10) : 24000;
      const wav = wrapPcmInWav(Buffer.from(inline.data, "base64"), rate);
      return { audioBase64: wav.toString("base64"), mimeType: "audio/wav" };
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError || new Error("speech unavailable");
}

/* ------------------------------ HTTP layer ------------------------------ */

function parseCookies(req) {
  const out = {};
  const header = req.headers.cookie || "";
  for (const part of header.split(";")) {
    const idx = part.indexOf("=");
    if (idx > 0) out[part.slice(0, idx).trim()] = part.slice(idx + 1).trim();
  }
  return out;
}

function sendJson(res, status, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(body);
}

function readBody(req, limitBytes) {
  // A serverless host may hand the function an already-parsed body
  // instead of a live stream; accept both shapes.
  if (req.body !== undefined && req.body !== null) {
    const text =
      typeof req.body === "string"
        ? req.body
        : Buffer.isBuffer(req.body)
          ? req.body.toString("utf8")
          : JSON.stringify(req.body);
    if (Buffer.byteLength(text) > limitBytes) {
      return Promise.reject(new Error("too large"));
    }
    return Promise.resolve(text);
  }
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > limitBytes) {
        reject(new Error("too large"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

async function handleRequest(req, res) {
  try {
    await store.ready();
    const url = new URL(req.url, "http://localhost");
    const pathname = url.pathname;

    // Visitor identity: anonymous random cookie, set on first API contact.
    const cookies = parseCookies(req);
    let visitor = cookies.mid;
    let setCookie = null;
    if (!visitor || !/^[a-f0-9-]{36}$/.test(visitor)) {
      visitor = crypto.randomUUID();
      setCookie = `mid=${visitor}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Lax`;
    }
    const withCookie = (status, obj) => {
      if (setCookie) res.setHeader("Set-Cookie", setCookie);
      sendJson(res, status, obj);
    };

    if (pathname === "/api/config" && req.method === "GET") {
      return withCookie(200, { configured: Boolean(GEMINI_API_KEY) });
    }

    if (pathname === "/api/chats" && req.method === "GET") {
      return withCookie(200, { chats: (await store.listChats(visitor)).map(toChat) });
    }
    if (pathname === "/api/chats" && req.method === "POST") {
      const chat = await store.createChat(visitor, "محادثة جديدة");
      return withCookie(200, { chat: toChat(chat) });
    }

    const chatMatch = pathname.match(/^\/api\/chats\/(\d+)$/);
    if (chatMatch && req.method === "DELETE") {
      await store.deleteChat(visitor, Number(chatMatch[1]));
      return withCookie(200, { ok: true });
    }

    const msgMatch = pathname.match(/^\/api\/chats\/(\d+)\/messages$/);
    if (msgMatch && req.method === "GET") {
      const chat = await store.getChat(visitor, Number(msgMatch[1]));
      if (!chat) return withCookie(404, { error: "المحادثة مش موجودة." });
      return withCookie(200, { messages: await Promise.all((await store.listMessages(chat.id)).map(toMessage)) });
    }
    if (msgMatch && req.method === "POST") {
      const chat = await store.getChat(visitor, Number(msgMatch[1]));
      if (!chat) return withCookie(404, { error: "المحادثة مش موجودة." });
      const body = JSON.parse((await readBody(req, 1_000_000)) || "{}");
      const text = String(body.text || "").trim().slice(0, 8000);
      if (!text) return withCookie(400, { error: "اكتب رسالة الأول." });
      const userMsg = await store.addMessage(chat.id, "user", "user", text, null);
      await store.touchChat(chat.id);
      if (!GEMINI_API_KEY) {
        const reply = await store.addMessage(
          chat.id,
          "assistant",
          "not_connected",
          "أنا ميليجي، ولسه بيتم تجهيزي من فريق Vision AI Studio. جرّب تاني كمان شوية.",
          null,
        );
        return withCookie(200, { userMessage: await toMessage(userMsg), reply: await toMessage(reply) });
      }
      // Automatic image intent: asking for a picture in plain chat
      // generates it right here. On failure, fall through to text.
      if (detectImageIntent(text)) {
        try {
          const { bytes, ext } = await geminiImage(text, []);
          const file = `${Date.now()}-${crypto.randomBytes(6).toString("hex")}.${ext}`;
          const imageRef = await files.put("images", file, bytes, imageMime(ext));
          const reply = await store.addMessage(chat.id, "assistant", "image", text, imageRef);
          return withCookie(200, { userMessage: await toMessage(userMsg), reply: await toMessage(reply) });
        } catch {
          /* fall through to the text reply */
        }
      }
      const history = (await store.listMessages(chat.id))
        .filter((m) => m.kind === "user" || m.role === "assistant")
        .slice(0, -1)
        .slice(-20)
        .map((m) => ({ role: m.role, content: m.content }));
      // Documents uploaded into this chat ride along as context.
      let modelText = text;
      try {
        const docs = await store.listDocuments(chat.id);
        let budget = 30000;
        const blocks = [];
        for (const doc of docs) {
          if (budget <= 0) break;
          const full = await store.getDocumentText(doc.id);
          if (!full) continue;
          const slice = full.slice(0, Math.min(12000, budget));
          budget -= slice.length;
          blocks.push(
            `[محتوى ملف «${doc.name}» اللي المستخدم رفعه في المحادثة دي — استخدمه للرد على أسئلته عنه بدقة، ومتخترعش محتوى مش موجود فيه]:\n${slice}`,
          );
        }
        if (blocks.length) modelText = `${text}\n\n${blocks.join("\n\n")}`;
      } catch {
        /* context is best-effort */
      }
      try {
        const { text: replyText, sources } = await geminiChat(history, modelText);
        let content = replyText;
        if (sources.length) {
          content += "\n\nالمصادر:\n" + sources.map((s) => `• ${s.title}: ${s.url}`).join("\n");
        }
        const reply = await store.addMessage(chat.id, "assistant", "user", content, null);
        return withCookie(200, { userMessage: await toMessage(userMsg), reply: await toMessage(reply) });
      } catch {
        const reply = await store.addMessage(
          chat.id,
          "assistant",
          "system_error",
          "حصلت مشكلة مؤقتة وأنا بحاول أرد. جرّب تبعت رسالتك تاني.",
          null,
        );
        return withCookie(200, { userMessage: await toMessage(userMsg), reply: await toMessage(reply) });
      }
    }

    const imgMatch = pathname.match(/^\/api\/chats\/(\d+)\/images$/);
    if (imgMatch && req.method === "POST") {
      const chat = await store.getChat(visitor, Number(imgMatch[1]));
      if (!chat) return withCookie(404, { error: "المحادثة مش موجودة." });
      const body = JSON.parse((await readBody(req, 24_000_000)) || "{}");
      const prompt = String(body.prompt || "").trim().slice(0, 2000);
      if (!prompt) return withCookie(400, { error: "اكتب وصف الصورة الأول." });
      const userMsg = await store.addMessage(chat.id, "user", "user", prompt, null);
      await store.touchChat(chat.id);
      if (!GEMINI_API_KEY) {
        const reply = await store.addMessage(
          chat.id,
          "assistant",
          "system_error",
          "توليد الصور لسه بيتم تجهيزه من فريق Vision AI Studio. جرّب تاني كمان شوية.",
          null,
        );
        return withCookie(200, { userMessage: await toMessage(userMsg), reply: await toMessage(reply) });
      }
      try {
        const { bytes, ext } = await geminiImage(prompt);
        const file = `${Date.now()}-${crypto.randomBytes(6).toString("hex")}.${ext}`;
        const imageRef = await files.put("images", file, bytes, imageMime(ext));
        const reply = await store.addMessage(chat.id, "assistant", "image", prompt, imageRef);
        return withCookie(200, { userMessage: await toMessage(userMsg), reply: await toMessage(reply) });
      } catch {
        const reply = await store.addMessage(
          chat.id,
          "assistant",
          "system_error",
          "مقدرتش أولّد الصورة دلوقتي. جرّب تاني كمان شوية.",
          null,
        );
        return withCookie(200, { userMessage: await toMessage(userMsg), reply: await toMessage(reply) });
      }
    }

    const docMatch = pathname.match(/^\/api\/chats\/(\d+)\/documents$/);
    if (docMatch && req.method === "POST") {
      const chat = await store.getChat(visitor, Number(docMatch[1]));
      if (!chat) return withCookie(404, { error: "المحادثة مش موجودة." });
      const body = JSON.parse((await readBody(req, 24_000_000)) || "{}");
      const fileName = String(body.fileName || "").trim().slice(0, 200);
      const dataBase64 = String(body.dataBase64 || "");
      const ext = (fileName.split(".").pop() || "").toLowerCase();
      const kindMeta = {
        pdf: { kind: "pdf", mime: "application/pdf" },
        docx: {
          kind: "docx",
          mime: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        },
        xlsx: {
          kind: "xlsx",
          mime: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        },
        txt: { kind: "txt", mime: "text/plain" },
        csv: { kind: "csv", mime: "text/csv" },
      }[ext];
      if (!kindMeta || !dataBase64) {
        return withCookie(400, {
          error: "نوع الملف ده مش مدعوم. ارفع ملف Word بصيغة docx أو Excel بصيغة xlsx أو PDF أو TXT أو CSV.",
        });
      }
      const file = `${Date.now()}-${crypto.randomBytes(6).toString("hex")}.${ext}`;
      const docRef = await files.put("documents", file, Buffer.from(dataBase64, "base64"), kindMeta.mime);
      let extractedText = "";
      try {
        extractedText = extractDocumentText(dataBase64, fileName).text.slice(0, 200000);
      } catch {
        extractedText = "";
      }
      const doc = await store.addDocument(
        chat.id,
        {
          name: fileName,
          kind: kindMeta.kind,
          mimeType: kindMeta.mime,
          file: docRef,
          charCount: extractedText.length,
          editedFromId: null,
        },
        extractedText,
      );
      const userMsg = await store.addMessage(chat.id, "user", "document", fileName, null, doc.id);
      await store.touchChat(chat.id);
      const readable = extractedText.trim().length >= 40;
      const ack = await store.addMessage(
        chat.id,
        "assistant",
        "user",
        readable
          ? `استلمت ملف «${fileName}» وقريته — حوالي ${extractedText.length.toLocaleString("ar-EG")} حرف. اسألني عن محتواه زي ما تحب، أو دوس «تعديل» على الملف وقولي عاوز تغيّر إيه بالظبط وهنفّذه وارجعلك نسخة معدّلة جاهزة للتنزيل.`
          : `استلمت ملف «${fileName}»، بس استخراج النص منه طلع محدود جدًا — ممكن يكون ملف مصوّر أو ممسوح ضوئيًا. هجاوب على قد ما أقدر من اللي قدرت أقراه، ولو عندك نسخة نصية منه ارفعها وهقراها كاملة.`,
        null,
      );
      return withCookie(200, { userMessage: await toMessage(userMsg), reply: await toMessage(ack) });
    }

    const editMatch = pathname.match(/^\/api\/documents\/(\d+)\/edit$/);
    if (editMatch && req.method === "POST") {
      const doc = await store.getDocument(Number(editMatch[1]));
      if (!doc) return withCookie(404, { error: "الملف ده مش موجود." });
      const chat = await store.getChat(visitor, doc.chat_id);
      if (!chat) return withCookie(404, { error: "الملف ده مش موجود." });
      const body = JSON.parse((await readBody(req, 100_000)) || "{}");
      const instruction = String(body.instruction || "").trim().slice(0, 2000);
      if (!instruction) return withCookie(400, { error: "اكتب التعديل اللي عاوزه الأول." });
      const originalText = await store.getDocumentText(doc.id);
      if (!originalText.trim()) {
        return withCookie(400, {
          error: "الملف ده مفيهوش نص مستخرج أقدر أعدّل عليه — جرّب ترفع نسخة نصية منه.",
        });
      }
      if (!GEMINI_API_KEY) {
        return withCookie(400, { error: "التعديل لسه بيتم تجهيزه من فريق Vision AI Studio." });
      }
      const isSheet = doc.kind === "xlsx" || doc.kind === "csv";
      const outKind = isSheet ? "xlsx" : doc.kind === "txt" ? "txt" : "docx";
      const formatRule = isSheet
        ? "المحتوى جدول بيانات: ارجع المحتوى المعدّل كله كصفوف مفصولة بعلامات تاب (Tab) بين الخلايا وسطر جديد بين الصفوف، من غير أي شرح أو عناوين إضافية أو أقواس كود."
        : "ارجع نص المستند المعدّل كله بس، فقرات مفصولة بسطور جديدة، من غير أي شرح أو تعليق أو أقواس كود أو علامات ماركداون.";
      try {
        const rawEdited = await geminiPlainText(
          "انت ميليجي، المساعد الذكي المصري، وطوّرتك شركة Vision AI Studio المصرية. مهمتك دلوقتي تحرير مستندات بدقة: تنفّذ تعليمات التعديل بالحرف وترجع المحتوى المعدّل بس.",
          `عدّل المستند التالي حسب تعليمات المستخدم بالحرف، وحافظ على باقي المحتوى زي ما هو من غير حذف أو اختصار إلا لو التعليمات طلبت كده.\n\nتعليمات المستخدم: ${instruction}\n\n${formatRule}\n\nنص المستند الأصلي:\n${originalText.slice(0, 60000)}`,
        );
        const edited = rawEdited
          .replace(/^```[a-z]*\n?/i, "")
          .replace(/\n?```$/i, "")
          .trim();
        if (!edited) throw new Error("empty edit");
        const baseName = doc.name.replace(/\.[^.]+$/, "");
        const built = buildDocumentFile(outKind, `${baseName} (معدّل)`, edited);
        const outExt = outKind;
        const file = `${Date.now()}-${crypto.randomBytes(6).toString("hex")}.${outExt}`;
        const docRef = await files.put("documents", file, Buffer.from(built.dataBase64, "base64"), built.mimeType);
        const newDoc = await store.addDocument(
          chat.id,
          {
            name: built.fileName,
            kind: outKind,
            mimeType: built.mimeType,
            file: docRef,
            charCount: edited.length,
            editedFromId: doc.id,
          },
          edited.slice(0, 200000),
        );
        const msg = await store.addMessage(
          chat.id,
          "assistant",
          "document",
          `النسخة المعدّلة من «${doc.name}» — نفّذت فيها: ${instruction.slice(0, 300)}`,
          null,
          newDoc.id,
        );
        await store.touchChat(chat.id);
        return withCookie(200, { message: await toMessage(msg) });
      } catch {
        return withCookie(400, { error: "مقدرتش أنفّذ التعديل دلوقتي، جرّب تاني بعد شوية." });
      }
    }

    if (pathname === "/api/transcribe" && req.method === "POST") {
      const body = JSON.parse((await readBody(req, 16_000_000)) || "{}");
      const audioBase64 = String(body.audioBase64 || "");
      const mimeType = String(body.mimeType || "audio/webm").split(";")[0];
      if (!audioBase64 || !mimeType.startsWith("audio/")) {
        return withCookie(400, { error: "الملف ده مش تسجيل صوتي." });
      }
      if (!GEMINI_API_KEY) {
        return withCookie(400, { error: "التفريغ الصوتي لسه بيتم تجهيزه من فريق Vision AI Studio." });
      }
      try {
        const text = await geminiTranscribe(audioBase64, mimeType);
        return withCookie(200, { text });
      } catch {
        return withCookie(400, {
          error: "مقدرتش أفهم التسجيل دلوقتي. جرّب تسجّل تاني في مكان هادي وبصوت واضح.",
        });
      }
    }

    if (pathname === "/api/speak" && req.method === "POST") {
      const body = JSON.parse((await readBody(req, 100_000)) || "{}");
      const text = String(body.text || "").trim().slice(0, 1500);
      if (!text) return withCookie(400, { error: "مفيش نص يتقال." });
      if (!GEMINI_API_KEY) {
        return withCookie(400, { error: "الصوت لسه بيتم تجهيزه من فريق Vision AI Studio." });
      }
      try {
        const result = await geminiSpeak(text);
        return withCookie(200, result);
      } catch {
        return withCookie(400, { error: "الصوت متولدش دلوقتي، جرّب تاني بعد شوية." });
      }
    }

    // Generated images and document files stored on local disk
    // (local development only — in production these are public Blob
    // URLs and never reach this route). The /api/media/* shape is
    // what the vercel.json rewrites point the public /images/* and
    // /documents/* paths at, so both shapes are answered here.
    const mediaMatch = pathname.match(
      /^\/(?:api\/media\/)?(images|documents)\/([^/]+)$/,
    );
    if (mediaMatch && req.method === "GET") {
      const kind = mediaMatch[1];
      const data = await files.get(kind, mediaMatch[2]);
      if (!data) return sendJson(res, 404, { error: "not found" });
      const ext = path.extname(mediaMatch[2]).toLowerCase();
      if (kind === "documents") {
        const mimeMap = {
          ".pdf": "application/pdf",
          ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
          ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          ".txt": "text/plain; charset=utf-8",
          ".csv": "text/csv; charset=utf-8",
        };
        res.writeHead(200, {
          "Content-Type": mimeMap[ext] || "application/octet-stream",
          "Content-Disposition": `attachment; filename="${mediaMatch[2]}"`,
        });
      } else {
        res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
      }
      return res.end(data);
    }

    // Static files (the UI). "/" serves index.html; anything else that is
    // not an API route falls back to index.html so the page always loads.
    let rel = pathname === "/" ? "index.html" : pathname.slice(1);
    let full = path.join(PUBLIC_DIR, rel);
    if (!full.startsWith(PUBLIC_DIR) || !fs.existsSync(full) || fs.statSync(full).isDirectory()) {
      full = path.join(PUBLIC_DIR, "index.html");
      rel = "index.html";
    }
    const ext = path.extname(full).toLowerCase();
    res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
    return res.end(fs.readFileSync(full));
  } catch (err) {
    console.error(err);
    if (!res.headersSent) sendJson(res, 500, { error: "حصلت مشكلة مؤقتة، جرّب تاني." });
  }
}

module.exports = { handleRequest };
