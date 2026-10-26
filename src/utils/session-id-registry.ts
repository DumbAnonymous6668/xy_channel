// A2A sessionId ↔ effective sessionId 持久化映射
//
// 背景：主对话的 A2A sessionId 固定为 "00000000"，前端永不更换，导致
// openclaw 侧无法靠"换 sessionId"天然隔离上下文（sessions.reset 只重置
// 会话元数据、打 reset 边界，并不会真正清空模型读到的上下文，且其前置
// harness 清理易被上下文压缩模型超时/报错连累而失败）。
//
// 方案：xy_channel 自己给每个前端会话维护一个"effective sessionId"（在
// 前端 sessionId 后追加一个随机后缀），并让它贯穿 xy_channel→openclaw
// 的会话路由（peer.id / conversationId / sessionStore）。配合
// dmScope="per-peer"，openclaw 会把不同 effective sessionId 当成不同会话，
// 从而在收到 clearContext 时只需轮换 effective sessionId，即可让 openclaw
// 视作全新上下文 —— 彻底绕开 sessions.reset，不受压缩模型故障影响。
//
// 存储位置与 cron-conv-map.json 同目录：/home/sandbox/.openclaw/session-id-map.json
import { promises as fs } from "fs";
import * as path from "path";
import { randomBytes } from "crypto";
import { logger } from "./logger.js";
import { withAsyncLock } from "./async-mutex.js";

const SESSION_ID_MAP_FILE = "/home/sandbox/.openclaw/session-id-map.json";

export interface SessionIdMapEntry {
  effectiveSessionId: string;
  createdAt: number;
  /** 最近一次轮换时间（首次创建时与 createdAt 相同）。 */
  updatedAt: number;
}

export interface SessionIdMapFile {
  version: 1;
  entries: Record<string, SessionIdMapEntry>;
}

/** 生成 12 位随机后缀（6 字节随机 hex，12 个字符），追加到前端 sessionId 之后。 */
export function generateRandomSuffix(): string {
  return randomBytes(6).toString("hex");
}

async function ensureDirectoryExists(filePath: string): Promise<void> {
  const dir = path.dirname(filePath);
  try {
    await fs.mkdir(dir, { recursive: true });
  } catch (error) {
    logger.error(`[SessionIdRegistry] Failed to create directory ${dir}:`, error);
  }
}

async function readMap(): Promise<SessionIdMapFile> {
  try {
    await ensureDirectoryExists(SESSION_ID_MAP_FILE);
    const content = await fs.readFile(SESSION_ID_MAP_FILE, "utf-8");
    const parsed = JSON.parse(content);
    if (
      parsed &&
      typeof parsed === "object" &&
      parsed.version === 1 &&
      parsed.entries &&
      typeof parsed.entries === "object"
    ) {
      return parsed as SessionIdMapFile;
    }
    logger.warn(`[SessionIdRegistry] Unexpected file shape, returning empty map`);
    return { version: 1, entries: {} };
  } catch (error: any) {
    if (error.code === "ENOENT") {
      return { version: 1, entries: {} };
    }
    logger.error(`[SessionIdRegistry] Failed to read map:`, error);
    return { version: 1, entries: {} };
  }
}

async function writeMap(map: SessionIdMapFile): Promise<void> {
  try {
    await ensureDirectoryExists(SESSION_ID_MAP_FILE);
    await fs.writeFile(SESSION_ID_MAP_FILE, JSON.stringify(map, null, 2), "utf-8");
  } catch (error) {
    logger.error(`[SessionIdRegistry] Failed to write map:`, error);
    throw error;
  }
}

/**
 * 计算满足 openclaw validateSessionId 正则（/^[a-z0-9][a-z0-9._-]{0,127}$/i）
 * 的 effective sessionId。前端 sessionId 通常是合法的 UUID，直接追加 "-后缀"
 * 即可；若前端 id 非法（含冒号等），退化为只使用随机后缀。
 */
export function buildEffectiveSessionId(frontendSessionId: string, suffix: string): string {
  const base = frontendSessionId || "default";
  if (/^[a-z0-9][a-z0-9._-]{0,127}$/i.test(base)) {
    const candidate = `${base}-${suffix}`;
    return candidate.length <= 128 ? candidate : suffix;
  }
  return suffix;
}

/**
 * 按前端 sessionId 取 effective sessionId：已有映射直接返回；无映射则
 * 生成一个持久化映射并落盘（首次建立）。失败时返回 null，不抛出，
 * 避免阻塞消息主流程（此时调用方应回退到原始前端 sessionId）。
 */
export async function getOrCreateSessionId(frontendSessionId: string): Promise<string | null> {
  if (!frontendSessionId || typeof frontendSessionId !== "string") {
    logger.warn(`[SessionIdRegistry] Invalid frontendSessionId: ${frontendSessionId}`);
    return null;
  }
  try {
    // 读→改→写 整体加互斥锁，防并发写覆盖丢映射。
    return await withAsyncLock(async () => {
      const map = await readMap();
      const existing = map.entries[frontendSessionId];
      if (existing?.effectiveSessionId) {
        return existing.effectiveSessionId;
      }
      const suffix = generateRandomSuffix();
      const effectiveSessionId = buildEffectiveSessionId(frontendSessionId, suffix);
      const now = Date.now();
      map.entries[frontendSessionId] = {
        effectiveSessionId,
        createdAt: now,
        updatedAt: now,
      };
      await writeMap(map);
      logger.log(
        `[SessionIdRegistry] Created effective sessionId for frontend=${frontendSessionId} -> ${effectiveSessionId}`,
      );
      return effectiveSessionId;
    });
  } catch (error) {
    logger.error(`[SessionIdRegistry] Failed to getOrCreateSessionId:`, error);
    return null;
  }
}

/**
 * 强制轮换 effective sessionId（clearContext）：无论原映射是否存在，
 * 都生成一个新的 effective sessionId 覆盖旧映射并持久化，让 openclaw
 * 视为全新会话（新上下文），从而真正清空。失败时返回 null，不抛出。
 */
export async function rotateSessionId(frontendSessionId: string): Promise<string | null> {
  if (!frontendSessionId || typeof frontendSessionId !== "string") {
    logger.warn(`[SessionIdRegistry] Invalid frontendSessionId: ${frontendSessionId}`);
    return null;
  }
  try {
    return await withAsyncLock(async () => {
      const map = await readMap();
      const suffix = generateRandomSuffix();
      const effectiveSessionId = buildEffectiveSessionId(frontendSessionId, suffix);
      const now = Date.now();
      const createdAt = map.entries[frontendSessionId]?.createdAt ?? now;
      map.entries[frontendSessionId] = {
        effectiveSessionId,
        createdAt,
        updatedAt: now,
      };
      await writeMap(map);
      logger.log(
        `[SessionIdRegistry] Rotated effective sessionId for frontend=${frontendSessionId} -> ${effectiveSessionId}`,
      );
      return effectiveSessionId;
    });
  } catch (error) {
    logger.error(`[SessionIdRegistry] Failed to rotateSessionId:`, error);
    return null;
  }
}
