/**
 * @fileoverview A library for non-AI based semantic search using TF-IDF and Cosine Similarity.
 * - findBestMatch: Finds the best matching document for a query from a list of documents.
 */

// Simple tokenizer: splits string into words, converts to lowercase, removes punctuation.
function tokenize(text: string): string[] {
    return text.toLowerCase().replace(/[.,!?]/g, '').split(/\s+/).filter(Boolean);
}

// Calculates Term Frequency (TF) for each word in a document.
function calculateTf(tokens: string[]): Map<string, number> {
    const tf = new Map<string, number>();
    const tokenCount = tokens.length;
    if (tokenCount === 0) return tf;

    for (const token of tokens) {
        tf.set(token, (tf.get(token) || 0) + 1);
    }

    for (const [token, count] of tf.entries()) {
        tf.set(token, count / tokenCount);
    }
    return tf;
}

// Calculates Inverse Document Frequency (IDF) for each word in the corpus.
function calculateIdf(documents: string[][]): Map<string, number> {
    const idf = new Map<string, number>();
    const docCount = documents.length;
    const wordInDocCount = new Map<string, number>();

    for (const doc of documents) {
        const uniqueTokens = new Set(doc);
        for (const token of uniqueTokens) {
            wordInDocCount.set(token, (wordInDocCount.get(token) || 0) + 1);
        }
    }

    for (const [token, count] of wordInDocCount.entries()) {
        idf.set(token, Math.log(docCount / (1 + count)));
    }
    return idf;
}

// Creates a TF-IDF vector for a document.
function createVector(tokens: string[], idf: Map<string, number>): Map<string, number> {
    const tf = calculateTf(tokens);
    const vector = new Map<string, number>();
    for (const [token, tfValue] of tf.entries()) {
        const idfValue = idf.get(token) || 0;
        vector.set(token, tfValue * idfValue);
    }
    return vector;
}

// Calculates the cosine similarity between two vectors.
function cosineSimilarity(vecA: Map<string, number>, vecB: Map<string, number>): number {
    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;

    const allTokens = new Set([...vecA.keys(), ...vecB.keys()]);

    for (const token of allTokens) {
        const valA = vecA.get(token) || 0;
        const valB = vecB.get(token) || 0;
        dotProduct += valA * valB;
    }

    for (const val of vecA.values()) {
        magnitudeA += val * val;
    }
    magnitudeA = Math.sqrt(magnitudeA);

    for (const val of vecB.values()) {
        magnitudeB += val * val;
    }
    magnitudeB = Math.sqrt(magnitudeB);

    if (magnitudeA === 0 || magnitudeB === 0) {
        return 0;
    }

    return dotProduct / (magnitudeA * magnitudeB);
}

/**
 * Finds the best matching document for a query from a list of documents.
 * @param query The user's input string.
 * @param documents An array of strings representing the documents to search through.
 * @returns An object containing the best match and all ratings.
 */
export function findBestMatch(query: string, documents: string[]): {
    ratings: { target: string, rating: number }[];
    bestMatch: { target: string, rating: number };
} {
    const tokenizedDocs = documents.map(tokenize);
    const tokenizedQuery = tokenize(query);

    const idf = calculateIdf(tokenizedDocs);

    const queryVector = createVector(tokenizedQuery, idf);
    const docVectors = tokenizedDocs.map(doc => createVector(doc, idf));

    const ratings = [];
    let bestMatch = { target: '', rating: -1 };

    for (let i = 0; i < documents.length; i++) {
        const similarity = cosineSimilarity(queryVector, docVectors[i]);
        const rating = { target: documents[i], rating: similarity };
        ratings.push(rating);
        if (similarity > bestMatch.rating) {
            bestMatch = rating;
        }
    }
    
    ratings.sort((a, b) => b.rating - a.rating);

    return { ratings, bestMatch };
}
