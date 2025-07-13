/**
 * @fileoverview A library for AI-based semantic search using TensorFlow.js and the Universal Sentence Encoder.
 * - findBestMatch: Finds the best matching document for a query from a list of documents.
 */
'use server';

import * as tf from '@tensorflow/tfjs-node';
import * as use from '@tensorflow-models/universal-sentence-encoder';

let model: use.UniversalSentenceEncoder | null = null;

async function loadModel() {
    if (!model) {
        console.log('Loading Universal Sentence Encoder model...');
        model = await use.load();
        console.log('Model loaded.');
    }
    return model;
}

// Calculates the cosine similarity between two vectors.
function cosineSimilarity(vecA: tf.Tensor, vecB: tf.Tensor): number {
    const dotProduct = tf.dot(vecA, vecB).dataSync()[0];
    const normA = tf.norm(vecA).dataSync()[0];
    const normB = tf.norm(vecB).dataSync()[0];

    if (normA === 0 || normB === 0) {
        return 0;
    }

    return dotProduct / (normA * normB);
}

/**
 * Finds the best matching document for a query from a list of documents using USE.
 * @param query The user's input string.
 * @param documents An array of strings representing the documents to search through.
 * @returns An object containing the best match and all ratings.
 */
export async function findBestMatch(query: string, documents: string[]): Promise<{
    ratings: { target: string, rating: number }[];
    bestMatch: { target: string, rating: number };
}> {
    const sentenceEncoder = await loadModel();
    
    const sentences = [query, ...documents];
    const embeddings = await sentenceEncoder.embed(sentences);

    const queryVector = embeddings.slice([0, 0], [1, -1]).squeeze();
    const docVectors = embeddings.slice([1, 0]);

    const ratings = [];
    let bestMatch = { target: '', rating: -1 };

    for (let i = 0; i < documents.length; i++) {
        const docVector = docVectors.slice([i, 0], [1, -1]).squeeze();
        const similarity = cosineSimilarity(queryVector, docVector);
        
        const rating = { target: documents[i], rating: similarity };
        ratings.push(rating);

        if (similarity > bestMatch.rating) {
            bestMatch = rating;
        }
        
        docVector.dispose();
    }

    ratings.sort((a, b) => b.rating - a.rating);
    
    queryVector.dispose();
    docVectors.dispose();
    embeddings.dispose();

    return { ratings, bestMatch };
}
