
// src/lib/idb.ts
import { openDB, type DBSchema } from 'idb';

const DB_NAME = 'WordWiseDB';
const DB_VERSION = 1;
const STORE_NAME = 'checkpoints';
const KEY = 'latest_checkpoint';

interface WordWiseDBSchema extends DBSchema {
  [STORE_NAME]: {
    key: string;
    value: string; // Storing serialized JSON as a string
  };
}

const dbPromise = openDB<WordWiseDBSchema>(DB_NAME, DB_VERSION, {
  upgrade(db) {
    if (!db.objectStoreNames.contains(STORE_NAME)) {
      db.createObjectStore(STORE_NAME);
    }
  },
});

export async function saveCheckpoint(modelJson: string): Promise<void> {
  try {
    const db = await dbPromise;
    await db.put(STORE_NAME, modelJson, KEY);
  } catch (error) {
    console.error("Failed to save checkpoint to IndexedDB:", error);
  }
}

export async function getCheckpoint(): Promise<string | null> {
  try {
    const db = await dbPromise;
    const checkpoint = await db.get(STORE_NAME, KEY);
    return checkpoint || null;
  } catch (error) {
    console.error("Failed to get checkpoint from IndexedDB:", error);
    return null;
  }
}

export async function clearCheckpoint(): Promise<void> {
  try {
    const db = await dbPromise;
    await db.delete(STORE_NAME, KEY);
  } catch (error) {
    console.error("Failed to clear checkpoint from IndexedDB:", error);
  }
}

    